
# ota_benchmarks_v4.py
# Warm-start post-update training from the trained pre-update model.
# Adds parameter-change ratio reporting.
import os
import json
import random
import math
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import io, zipfile, urllib.request

try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None

try:
    import torchaudio
except Exception:
    torchaudio = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_default():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def prepare_ucihar(root: str):
    needed = [os.path.join(root, f) for f in ["X_train.npy","y_train.npy","X_test.npy","y_test.npy"]]
    if all(os.path.isfile(p) for p in needed):
        print("[UCIHAR] Found preprocessed .npy files.")
        return
    os.makedirs(root, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    print("[UCIHAR] Downloading dataset zip...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    z = zipfile.ZipFile(io.BytesIO(data))
    base = "UCI HAR Dataset/"
    signals = ["body_acc_x_","body_acc_y_","body_acc_z_",
               "body_gyro_x_","body_gyro_y_","body_gyro_z_",
               "total_acc_x_","total_acc_y_","total_acc_z_"]
    def load_txt(zf, path):
        with zf.open(path, "r") as f:
            return np.loadtxt(f)
    X_out, y_out = {}, {}
    for split in ["train","test"]:
        y = load_txt(z, base + f"{split}/y_{split}.txt").astype(np.int64) - 1
        chans = []
        for sig in signals:
            a = load_txt(z, base + f"{split}/Inertial Signals/{sig}{split}.txt")  # [N,128]
            a = a[:, :, None]  # -> [N,128,1]
            chans.append(a)
        X = np.concatenate(chans, axis=2)  # [N,128,9]
        X_out[split], y_out[split] = X, y
    np.save(os.path.join(root,"X_train.npy"), X_out["train"])
    np.save(os.path.join(root,"y_train.npy"), y_out["train"])
    np.save(os.path.join(root,"X_test.npy" ), X_out["test"])
    np.save(os.path.join(root,"y_test.npy"), y_out["test"])
    print("[UCIHAR] Built X/y .npy files at", root)
    

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def save_deltas(pre_state: Dict[str, torch.Tensor],
                post_state: Dict[str, torch.Tensor],
                out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for k in post_state.keys():
        if k not in pre_state:
            delta = post_state[k].cpu().numpy()
            path = os.path.join(out_dir, f"delta_NEW_{k.replace('.', '_')}.npy")
            np.save(path, delta)
            manifest.append({"name": k, "type": "new", "shape": list(delta.shape), "file": os.path.basename(path)})
        else:
            pre = pre_state[k].cpu().numpy()
            post = post_state[k].cpu().numpy()
            delta = post - pre
            if np.any(delta != 0):
                path = os.path.join(out_dir, f"delta_{k.replace('.', '_')}.npy")
                np.save(path, delta)
                manifest.append({"name": k, "type": "delta", "shape": list(delta.shape), "file": os.path.basename(path)})
    for k in pre_state.keys():
        if k not in post_state:
            manifest.append({"name": k, "type": "removed"})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def compute_change_ratio(pre_state: Dict[str, torch.Tensor],
                         post_state: Dict[str, torch.Tensor],
                         eps: float = 1e-8):
    total = 0
    changed = 0
    per_tensor = []
    for k, post in post_state.items():
        post_np = post.detach().cpu().numpy()
        numel = post_np.size
        total += numel
        if k not in pre_state:
            # new tensor, count fully changed
            changed += numel
            per_tensor.append({"name": k, "numel": int(numel), "changed": int(numel), "reason": "new"})
        else:
            pre_np = pre_state[k].detach().cpu().numpy()
            diff = np.abs(post_np - pre_np) > eps
            c = int(diff.sum())
            changed += c
            per_tensor.append({"name": k, "numel": int(numel), "changed": c, "reason": "delta"})
    ratio = changed / max(1, total)
    return {"total_params": int(total), "changed_params": int(changed), "ratio": float(ratio), "details": per_tensor}


# -------- Models --------

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, extra_conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.extra_conv = extra_conv
        if extra_conv:
            self.conv2b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # dynamic fc1 in_features
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            if extra_conv:
                x = F.relu(self.conv2b(x))
            flat = x.numel()
        self.fc1 = nn.Linear(flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if hasattr(self, "conv2b"):
            x = F.relu(self.conv2b(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HARNET(nn.Module):
    def __init__(self, num_classes=6, channels=9, remove_mid=False, kernel_change: Optional[int] = None):
        super().__init__()
        k1 = 5
        k2 = 5 if kernel_change is None else kernel_change
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=k1, padding=k1 // 2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = None if remove_mid else nn.Conv1d(32, 64, kernel_size=k2, padding=k2 // 2)
        self.bn2 = None if remove_mid else nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64 if not remove_mid else 32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.conv2 is not None:
            x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class TinyKWS(nn.Module):
    def __init__(self, num_classes=10, widen=False):
        super().__init__()
        c1 = 16 if not widen else 20
        c2 = 32 if not widen else 40
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.head = nn.Linear(c2 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# -------- Datasets --------

class UCIHAR(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        x = np.load(os.path.join(root, f"X_{split}.npy"))
        y = np.load(os.path.join(root, f"y_{split}.npy"))
        x = np.transpose(x, (0, 2, 1))
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SpeechCommandsSubset(Dataset):
    def __init__(self, root: str, subset_labels: List[str], split: str = "train"):
        assert torchaudio is not None, "torchaudio is required for Speech Commands"
        self.labels = subset_labels
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        base = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset=split)
        self.items = []
        for waveform, sample_rate, label, *_ in base:
            if label in self.label_to_idx:
                self.items.append((waveform, sample_rate, self.label_to_idx[label]))
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=32)
        self.amptodb = torchaudio.transforms.AmplitudeToDB()
        self.fixed_len = 16000

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        waveform, sr, y = self.items[idx]
        if waveform.size(-1) < self.fixed_len:
            pad = self.fixed_len - waveform.size(-1)
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.fixed_len]
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        mel = self.amptodb(self.mel(waveform)).unsqueeze(0)
        mel = F.interpolate(mel, size=(32, 32), mode="bilinear", align_corners=False).squeeze(0)
        return mel, torch.tensor(y, dtype=torch.long)


def make_mnist(root: str, subset_ratio: float, batch_size: int, seed: int):
    assert torchvision is not None, "torchvision is required for MNIST"
    tf = T.Compose([T.ToTensor()])
    train = torchvision.datasets.MNIST(root=root, train=True, transform=tf, download=True)
    test = torchvision.datasets.MNIST(root=root, train=False, transform=tf, download=True)
    n = len(train)
    m = int(n * subset_ratio)
    g = torch.Generator().manual_seed(seed)
    subset_indices = torch.randperm(n, generator=g)[:m].tolist()
    pre_train = Subset(train, subset_indices)
    pre_loader = DataLoader(pre_train, batch_size=batch_size, shuffle=True, num_workers=2)
    post_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    return pre_loader, post_loader, test_loader


def make_ucihar(root: str, subset_ratio: float, batch_size: int, seed: int):
    prepare_ucihar(root)
    train = UCIHAR(root, split="train")
    test = UCIHAR(root, split="test")
    n = len(train)
    m = int(n * subset_ratio)
    g = torch.Generator().manual_seed(seed)
    subset_indices = torch.randperm(n, generator=g)[:m].tolist()
    pre_train = Subset(train, subset_indices)
    pre_loader = DataLoader(pre_train, batch_size=batch_size, shuffle=True, num_workers=2)
    post_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    return pre_loader, post_loader, test_loader


def make_kws(root: str, subset_ratio: float, batch_size: int, seed: int, labels: List[str]):
    train = SpeechCommandsSubset(root=root, subset_labels=labels, split="training")
    test = SpeechCommandsSubset(root=root, subset_labels=labels, split="testing")
    n = len(train)
    m = int(n * subset_ratio)
    g = torch.Generator().manual_seed(seed)
    subset_indices = torch.randperm(n, generator=g)[:m].tolist()
    pre_train = Subset(train, subset_indices)
    pre_loader = DataLoader(pre_train, batch_size=batch_size, shuffle=True, num_workers=2)
    post_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    return pre_loader, post_loader, test_loader


# Warm-start helpers

def center_copy_kernel(dst: torch.Tensor, src: torch.Tensor):
    dst.zero_()
    if dst.dim() == 3:
        _, _, kd = dst.shape
        _, _, ks = src.shape
        start = (kd - ks) // 2
        dst[..., start:start+ks] = src
    elif dst.dim() == 4:
        _, _, khd, kwd = dst.shape
        _, _, khs, kws = src.shape
        st_h = (khd - khs) // 2
        st_w = (kwd - kws) // 2
        dst[..., st_h:st_h+khs, st_w:st_w+kws] = src
    return dst


def partial_load_from_pre(pre_state: Dict[str, torch.Tensor],
                          post_model: nn.Module,
                          scenario: str):
    post_state = post_model.state_dict()
    transferred = []
    for k, v in post_state.items():
        if k in pre_state and pre_state[k].shape == v.shape:
            v.copy_(pre_state[k])
            transferred.append(k)
        else:
            if scenario in ["C8"] and ("conv1.weight" in k or "conv2.weight" in k):
                pre_k = k
                if pre_k in pre_state:
                    src = pre_state[pre_k]
                    dst = v
                    min_shape = tuple(min(a, b) for a, b in zip(dst.shape, src.shape))
                    slices = tuple(slice(0, m) for m in min_shape)
                    dst[slices] = src[slices]
                    transferred.append(k + " (partial widen)")
            elif scenario in ["C6"] and k == "conv2.weight":
                if k in pre_state:
                    dst = v
                    src = pre_state[k]
                    center_copy_kernel(dst, src)
                    transferred.append(k + " (kernel center copy)")
            elif scenario in ["C9"] and k in ["head.weight", "head.bias"]:
                if k in pre_state:
                    src = pre_state[k]
                    dst = v
                    rows = min(dst.shape[0], src.shape[0])
                    dst[:rows] = src[:rows]
                    transferred.append(k + " (head resize rows)")
    post_model.load_state_dict(post_state)
    return transferred


# Scenario mapping and scopes

def build_models_for_scenario(benchmark: str, scenario: str, num_classes_kws: int, seed: int):
    if benchmark == "B1":
        if scenario == "C1":
            pre = LeNet5(num_classes=10, extra_conv=False)
            post = LeNet5(num_classes=10, extra_conv=False)
            desc = "LeNet-5 weights refinement"
        elif scenario == "C2":
            pre = LeNet5(num_classes=10, extra_conv=False)
            post = LeNet5(num_classes=10, extra_conv=True)
            desc = "LeNet-5 add a small conv layer (conv2b)"
        elif scenario == "C3":
            pre = LeNet5(num_classes=10, extra_conv=False)
            post = LeNet5(num_classes=10, extra_conv=False)
            desc = "LeNet-5 head-only update"
        else:
            raise ValueError("Invalid scenario for B1")
    elif benchmark == "B2":
        if scenario == "C4":
            pre = HARNET(num_classes=6)
            post = HARNET(num_classes=6)
            desc = "HAR-Net layer-slice update"
        elif scenario == "C5":
            pre = HARNET(num_classes=6, remove_mid=False)
            post = HARNET(num_classes=6, remove_mid=True)
            desc = "HAR-Net remove mid conv layer"
        elif scenario == "C6":
            pre = HARNET(num_classes=6, kernel_change=None)
            post = HARNET(num_classes=6, kernel_change=9)
            desc = "HAR-Net change kernel length"
        else:
            raise ValueError("Invalid scenario for B2")
    elif benchmark == "B3":
        if scenario == "C7":
            pre = TinyKWS(num_classes=num_classes_kws, widen=False)
            post = TinyKWS(num_classes=num_classes_kws, widen=False)
            desc = "Tiny KWS bias-only update"
        elif scenario == "C8":
            pre = TinyKWS(num_classes=num_classes_kws, widen=False)
            post = TinyKWS(num_classes=num_classes_kws, widen=True)
            desc = "Tiny KWS widen conv channels"
        elif scenario == "C9":
            pre = TinyKWS(num_classes=num_classes_kws, widen=False)
            post = TinyKWS(num_classes=num_classes_kws + 1, widen=False)
            desc = "Tiny KWS head resize add 1 class"
        else:
            raise ValueError("Invalid scenario for B3")
    else:
        raise ValueError("Unknown benchmark")
    set_seed(seed)
    return pre, post, desc


def apply_training_scope(model: nn.Module, scenario: str):
    if scenario in ["C1", "C2", "C5", "C6", "C8"]:
        for p in model.parameters():
            p.requires_grad_(True)
    elif scenario == "C3":
        freeze_all(model)
        for name, p in model.named_parameters():
            if name.startswith("fc3"):
                p.requires_grad_(True)
    elif scenario == "C4":
        freeze_all(model)
        for name, p in model.named_parameters():
            if name.startswith("conv2"):
                p.requires_grad_(True)
    elif scenario == "C7":
        freeze_all(model)
        for name, p in model.named_parameters():
            if name.endswith(".bias") or "bias" in name:
                p.requires_grad_(True)
    elif scenario == "C9":
        freeze_all(model)
        for name, p in model.named_parameters():
            if name.startswith("head"):
                p.requires_grad_(True)
    else:
        raise ValueError("Unknown scenario")


def run_experiment(args):
    set_seed(args.seed)
    device = device_default()

    pre_model, post_model, desc = build_models_for_scenario(args.benchmark, args.scenario, args.kws_classes, args.seed)
    pre_model.to(device)
    post_model.to(device)

    if args.benchmark == "B1":
        pre_loader, post_loader, test_loader = make_mnist(args.data_root, args.pre_ratio, args.batch_size, args.seed)
    elif args.benchmark == "B2":
        pre_loader, post_loader, test_loader = make_ucihar(args.ucihar_root, args.pre_ratio, args.batch_size, args.seed)
    elif args.benchmark == "B3":
        labels = args.kws_label_list.split(",")
        assert len(labels) == args.kws_classes, "kws_classes must match kws_label_list count"
        pre_loader, post_loader, test_loader = make_kws(args.data_root, args.pre_ratio, args.batch_size, args.seed, labels)
    else:
        raise ValueError("Unknown benchmark")

    criterion = nn.CrossEntropyLoss()

    # Pre-update
    apply_training_scope(pre_model, args.scenario)
    opt_pre = torch.optim.Adam(filter(lambda p: p.requires_grad, pre_model.parameters()), lr=args.lr)
    for epoch in range(args.epochs_pre):
        tr_loss, tr_acc = train_one_epoch(pre_model, pre_loader, criterion, opt_pre, device)
        te_loss, te_acc = evaluate(pre_model, test_loader, criterion, device)
        print(f"[Pre] Epoch {epoch+1}/{args.epochs_pre} loss={tr_loss:.4f} val_acc={te_acc:.4f}")
    pre_state = {k: v.detach().cpu().clone() for k, v in pre_model.state_dict().items()}

    # Warm-start post
    transferred = partial_load_from_pre(pre_state, post_model, args.scenario)
    if len(transferred) > 0:
        print(f"Warm-started tensors: {len(transferred)}")
    else:
        print("Warm-started tensors: 0")

    # Post-update
    apply_training_scope(post_model, args.scenario)
    opt_post = torch.optim.Adam(filter(lambda p: p.requires_grad, post_model.parameters()), lr=args.lr)
    for epoch in range(args.epochs_post):
        tr_loss, tr_acc = train_one_epoch(post_model, post_loader, criterion, opt_post, device)
        te_loss, te_acc = evaluate(post_model, test_loader, criterion, device)
        print(f"[Post] Epoch {epoch+1}/{args.epochs_post} loss={tr_loss:.4f} val_acc={te_acc:.4f}")
    post_state = {k: v.detach().cpu().clone() for k, v in post_model.state_dict().items()}

    out_dir = os.path.join(args.out_dir, f"{args.benchmark}_{args.scenario}")
    os.makedirs(out_dir, exist_ok=True)
    manifest = save_deltas(pre_state, post_state, out_dir)

    # Change ratio stats
    stats = compute_change_ratio(pre_state, post_state, eps=1e-8)
    with open(os.path.join(out_dir, "change_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "scenario": args.scenario,
            "desc": desc,
            "epochs_pre": args.epochs_pre,
            "epochs_post": args.epochs_post,
            "pre_ratio": args.pre_ratio,
            "params_pre": count_parameters(pre_model),
            "params_post": count_parameters(post_model),
            "manifest_count": len(manifest),
            "warm_start_transferred_tensors": len([1 for _ in stats["details"] if _["reason"] in ("delta","new")]),
            "change_ratio": stats["ratio"],
            "changed_params": stats["changed_params"],
            "total_params": stats["total_params"]
        }, f, indent=2)

    print(f"Experiment complete. Deltas and manifest saved to: {out_dir}")
    print(f"Parameter-change ratio (post vs pre): {stats['changed_params']} / {stats['total_params']} = {stats['ratio']:.6f}")


def build_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=["B1", "B2", "B3"], required=True)
    parser.add_argument("--scenario", type=str, choices=[f"C{i}" for i in range(1, 10)], required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--ucihar_root", type=str, default="./data/UCIHAR")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs_pre", type=int, default=2)
    parser.add_argument("--epochs_post", type=int, default=2)
    parser.add_argument("--pre_ratio", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kws_classes", type=int, default=10)
    parser.add_argument("--kws_label_list", type=str, default="yes,no,up,down,left,right,on,off,stop,go")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run_experiment(args)
