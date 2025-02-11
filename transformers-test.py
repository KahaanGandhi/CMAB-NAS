#!/usr/bin/env python3
"""
Bandit-based NAS for Vision Transformers on CIFAR-10
"""

import os, random, math, copy, time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

# ====== Enhanced CIFAR-10 Data Loading with Strong Augmentation ======

def get_cifar10_loaders(batch_size=96, num_workers=2, cutout_length=16):
    mean = (0.49139968, 0.48215827, 0.44653124)
    std  = (0.24703233, 0.24348505, 0.26158768)
    
    # Training augmentations: random crop, horizontal flip, RandAugment, normalization, Cutout and RandomErasing.
    transform_train_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if cutout_length > 0:
        transform_train_list.append(CutoutDefault(cutout_length))
    transform_train = transforms.Compose(transform_train_list)
    transform_train = transforms.Compose([
        transform_train,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    
    if os.name == "nt":
        train_workers = 0
        val_workers = 0
    else:
        train_workers = num_workers
        val_workers = num_workers

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=train_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
                                             num_workers=val_workers, pin_memory=True)
    return train_loader, val_loader

class CutoutDefault(object):
    """Cutout augmentation: randomly masks out a square patch."""
    def __init__(self, length):
        self.length = length
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img

# ====== Transformer Search Space Candidate Lists ======

CANDIDATE_ATTN = ["global", "window", "shifted_window"]
CANDIDATE_FFN = ["ffn_2", "ffn_4", "ffn_8"]

# ====== Transformer Candidate Operations ======

# Global Self-Attention using nn.MultiheadAttention (batch_first=True)
class GlobalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GlobalSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        return attn_out

# Windowed Self-Attention: partition tokens into non-overlapping windows
class WindowSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=4, dropout=0.1):
        super(WindowSelfAttention, self).__init__()
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x):
        B, N, E = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, E)
        ws = self.window_size
        num_windows_h = H // ws
        num_windows_w = W // ws
        x_windows = x.view(B, num_windows_h, ws, num_windows_w, ws, E)
        x_windows = x_windows.permute(0,1,3,2,4,5).contiguous().view(-1, ws*ws, E)
        attn_out, _ = self.mha(x_windows, x_windows, x_windows)
        attn_out = attn_out.view(B, num_windows_h, num_windows_w, ws, ws, E)
        attn_out = attn_out.permute(0,1,3,2,4,5).contiguous().view(B, H*W, E)
        return attn_out

# Shifted Window Self-Attention: applies cyclic shift before window partitioning
class ShiftedWindowSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=4, shift_size=None, dropout=0.1):
        super(ShiftedWindowSelfAttention, self).__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift_size is None else shift_size
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x):
        B, N, E = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, E)
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        ws = self.window_size
        num_windows_h = H // ws
        num_windows_w = W // ws
        x_windows = shifted_x.view(B, num_windows_h, ws, num_windows_w, ws, E)
        x_windows = x_windows.permute(0,1,3,2,4,5).contiguous().view(-1, ws*ws, E)
        attn_out, _ = self.mha(x_windows, x_windows, x_windows)
        attn_out = attn_out.view(B, num_windows_h, num_windows_w, ws, ws, E)
        attn_out = attn_out.permute(0,1,3,2,4,5).contiguous().view(B, H, W, E)
        out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        out = out.view(B, H*W, E)
        return out

def create_attn_op(name, embed_dim, num_heads=8, dropout=0.1):
    if name == "global":
        return GlobalSelfAttention(embed_dim, num_heads, dropout)
    elif name == "window":
        return WindowSelfAttention(embed_dim, num_heads, window_size=4, dropout=dropout)
    elif name == "shifted_window":
        return ShiftedWindowSelfAttention(embed_dim, num_heads, window_size=4, dropout=dropout)
    else:
        raise ValueError(f"Unknown attention candidate: {name}")

# FFN block with variable expansion factor
class FFNBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor, dropout_rate=0.1):
        super(FFNBlock, self).__init__()
        hidden_dim = embed_dim * expansion_factor
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

def create_ffn_op(name, embed_dim, dropout=0.1):
    if name == "ffn_2":
        return FFNBlock(embed_dim, expansion_factor=2, dropout_rate=dropout)
    elif name == "ffn_4":
        return FFNBlock(embed_dim, expansion_factor=4, dropout_rate=dropout)
    elif name == "ffn_8":
        return FFNBlock(embed_dim, expansion_factor=8, dropout_rate=dropout)
    else:
        raise ValueError(f"Unknown FFN candidate: {name}")

# ====== Mixed Transformer Cell (Pre-Norm Residual Block) ======

class MixedTransformerCell(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(MixedTransformerCell, self).__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Candidate self-attention operations (weight-sharing)
        self.attn_ops = nn.ModuleDict({
            key: create_attn_op(key, embed_dim, num_heads=8, dropout=dropout) for key in CANDIDATE_ATTN
        })
        # Candidate FFN operations
        self.ffn_ops = nn.ModuleDict({
            key: create_ffn_op(key, embed_dim, dropout=dropout) for key in CANDIDATE_FFN
        })
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, arch):
        # Ensure arch is a tuple (attn_choice, ffn_choice); if provided as list, take first tuple.
        if isinstance(arch, (list, tuple)) and (not isinstance(arch[0], str)):
            arch = arch[0]
        attn_choice, ffn_choice = arch
        # Self-Attention sub-block (pre-norm)
        x_norm = self.norm1(x)
        attn_out = self.attn_ops[attn_choice](x_norm)
        x = x + self.dropout(attn_out)
        # FFN sub-block (pre-norm)
        x_norm = self.norm2(x)
        ffn_out = self.ffn_ops[ffn_choice](x_norm)
        x = x + self.dropout(ffn_out)
        return x

# ====== Local MAB for Transformer Cells ======

class LocalMAB_Transformer:
    """
    Local multi-armed bandit: each arm is a tuple (attn_choice, ffn_choice).
    """
    def __init__(self, candidate_attn, candidate_ffn):
        self.arms = []
        for a in candidate_attn:
            for f in candidate_ffn:
                self.arms.append((a, f))
        self.arm_info = [dict(total_reward=0.0, num_plays=0) for _ in self.arms]
        self.num_plays_global = 0
    def ucb_select_arm(self, alpha=1.0):
        best_idx = 0
        best_val = -1e9
        for idx, rec in enumerate(self.arm_info):
            n_j = rec['num_plays']
            if n_j == 0:
                avg_r = 0.0
                bonus = 1e6  # force exploration
            else:
                avg_r = rec['total_reward'] / n_j
                bonus = alpha * math.sqrt(2.0 * math.log(self.num_plays_global + 1e-12) / (n_j + 1e-12))
            val = avg_r + bonus
            if val > best_val:
                best_val = val
                best_idx = idx
        return best_idx
    def record_reward(self, arm, reward):
        idx = self.arms.index(arm)
        self.num_plays_global += 1
        self.arm_info[idx]['total_reward'] += reward
        self.arm_info[idx]['num_plays'] += 1
    def best_arm(self):
        best_idx = 0
        best_avg = -1e9
        for idx, rec in enumerate(self.arm_info):
            if rec['num_plays'] == 0:
                continue
            avg_r = rec['total_reward'] / rec['num_plays']
            if avg_r > best_avg:
                best_avg = avg_r
                best_idx = idx
        return self.arms[best_idx]

class NMCSCellSearch_Transformer:
    def __init__(self, num_cells, candidate_attn, candidate_ffn, alpha=1.0):
        self.alpha = alpha
        self.num_cells = num_cells
        self.local_mabs = [LocalMAB_Transformer(candidate_attn, candidate_ffn) for _ in range(num_cells)]
    def sample_architecture_ucb(self):
        arch = []
        for mab in self.local_mabs:
            idx = mab.ucb_select_arm(alpha=self.alpha)
            arch.append(mab.arms[idx])
        return arch
    def record_reward(self, arch, reward):
        share = reward / self.num_cells
        for i, arm in enumerate(arch):
            self.local_mabs[i].record_reward(arm, share)
    def best_architecture_local_optimal(self):
        arch = []
        for mab in self.local_mabs:
            arm = mab.best_arm()
            arch.append(arm)
        return arch

# ====== Proxy Transformer Network with Weight Sharing ======

class ProxyTransformerNetwork(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=8, num_classes=10):
        super(ProxyTransformerNetwork, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_patches = (img_size // patch_size) ** 2
        # Patch embedding layer: a conv layer with kernel/stride equal to patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)
        # Stack transformer cells
        self.cells = nn.ModuleList([MixedTransformerCell(embed_dim, dropout=0.1) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x, arch_config):
        # arch_config: list of length 'depth' with a tuple (attn_choice, ffn_choice) per cell
        B = x.size(0)
        x = self.patch_embed(x)   # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        for i, cell in enumerate(self.cells):
            # Use the i-th configuration; if not provided, default to ("global", "ffn_4")
            cell_arch = arch_config[i] if arch_config is not None else ("global", "ffn_4")
            x = cell(x, cell_arch)
        x = self.norm(x)
        # Global average pooling over tokens
        x = x.mean(dim=1)
        x = self.head(x)
        return x

# ====== TransformerBanditNAS Orchestrator ======

class TransformerBanditNAS:
    """
    Orchestrates transformer cell architecture search via bandit strategies.
    Uses NMCSCellSearch_Transformer for local decisions and a weight-sharing proxy network.
    """
    def __init__(self, device, alpha=1.0, lr=0.001, weight_decay=0.05,
                 epochs=50, batch_size=96, depth=8):
        self.device = device
        self.alpha = alpha
        self.epochs = epochs
        self.depth = depth
        self.nmcs_transformer = NMCSCellSearch_Transformer(num_cells=depth,
                                                           candidate_attn=CANDIDATE_ATTN,
                                                           candidate_ffn=CANDIDATE_FFN,
                                                           alpha=alpha)
        self.proxy = ProxyTransformerNetwork(img_size=32, patch_size=4, embed_dim=192,
                                               depth=depth, num_classes=10).to(device)
        self.optimizer = optim.AdamW(self.proxy.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.val_loader = get_cifar10_loaders(batch_size=batch_size, cutout_length=16)
        self.best_arch = None
        self.best_acc = 0.0
        self.warmup_done = False
        self.current_epoch = 0
        self.batch_size = batch_size
    def _warmup_proxy(self, warmup_epochs):
        if self.warmup_done:
            print("[Warmup] Already done, skipping.")
            return
        print(f"[Warmup] Running {warmup_epochs} warmup epochs (random sampling).")
        for ep in range(warmup_epochs):
            total_loss = 0.0
            total_inst = 0
            correct = 0
            self.proxy.train()
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                arch = self.nmcs_transformer.sample_architecture_ucb()
                self.optimizer.zero_grad()
                logits = self.proxy(inputs, arch)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                _, pred = logits.max(1)
                total_inst += inputs.size(0)
                correct += pred.eq(targets).sum().item()
            avg_loss = total_loss / total_inst
            acc = 100.0 * correct / total_inst
            print(f"  Warmup Epoch {ep+1}/{warmup_epochs} -- Loss: {avg_loss:.3f}, Acc: {acc:.2f}%")
        self.warmup_done = True
    def _train_child_architecture(self, arch, train_iter):
        self.proxy.train()
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(self.train_loader)
            inputs, targets = next(train_iter)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        logits = self.proxy(inputs, arch)
        loss = self.criterion(logits, targets)
        loss.backward()
        # Apply gradient clipping for transformer stability
        torch.nn.utils.clip_grad_norm_(self.proxy.parameters(), max_norm=5.0)
        self.optimizer.step()
        return train_iter
    def _eval_child_architecture(self, arch, num_batches=1):
        self.proxy.eval()
        correct = 0
        total = 0
        val_iter = iter(self.val_loader)
        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    inputs, targets = next(val_iter)
                except StopIteration:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                out = self.proxy(inputs, arch)
                _, pred = out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        return 100.0 * correct / total if total > 0 else 0.0
    def _eval_full_architecture(self, arch):
        self.proxy.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                out = self.proxy(inputs, arch)
                _, pred = out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        return 100.0 * correct / total if total > 0 else 0.0
    def search(self, B=2500, warmup_epochs=5):
        # B: number of candidate child architectures per epoch.
        if not self.warmup_done:
            self._warmup_proxy(warmup_epochs)
        L = 8  # Simulation iterations to update local bandit rewards.
        train_iter = iter(self.train_loader)
        for ep in range(self.epochs):
            self.current_epoch = ep
            print(f"\n=== Search Epoch {ep+1}/{self.epochs} ===")
            print(f"  (Alpha: {self.nmcs_transformer.alpha:.3f})")
            # Simulation: update local rewards with short evaluations.
            for sim in range(L):
                sim_arch = self.nmcs_transformer.sample_architecture_ucb()
                sim_reward = self._eval_child_architecture(sim_arch, num_batches=1)
                self.nmcs_transformer.record_reward(sim_arch, sim_reward)
            candidate_archs = []
            for _ in range(B):
                arch = self.nmcs_transformer.sample_architecture_ucb()
                candidate_archs.append(arch)
            for i in tqdm(range(B), desc=f"Epoch {ep+1}/{self.epochs} - Batch Progress", leave=False):
                arch = candidate_archs[i]
                train_iter = self._train_child_architecture(arch, train_iter)
                reward = self._eval_child_architecture(arch, num_batches=1)
                self.nmcs_transformer.record_reward(arch, reward)
            self.nmcs_transformer.alpha *= 0.95
            best_arch = self.nmcs_transformer.best_architecture_local_optimal()
            acc_opt = self._eval_full_architecture(best_arch)
            print(f"  Local-opt architecture validation accuracy: {acc_opt:.2f}%")
            if acc_opt > self.best_acc:
                self.best_acc = acc_opt
                self.best_arch = copy.deepcopy(best_arch)
                print(f"  [Update] New best architecture found: {acc_opt:.2f}%")
        print(f"\n=== SEARCH FINISHED ===")
        print(f"Best discovered architecture validation accuracy: {self.best_acc:.2f}%")
        return self.best_arch

# ====== Final Discovered Transformer Network ======

class DiscoveredTransformerNetwork(nn.Module):
    """
    Stand-alone transformer network built from the best discovered cell configuration.
    This final network uses 20 cells. For cells, we cycle through the discovered configuration
    if the number of cells exceeds the proxy depth.
    """
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=20, num_classes=10, best_arch=None):
        super(DiscoveredTransformerNetwork, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        if best_arch is None:
            self.best_arch = [("global", "ffn_4")] * 8  # default configuration if none discovered
        else:
            self.best_arch = best_arch  # best_arch is expected as a list from the search phase.
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.cells = nn.ModuleList([MixedTransformerCell(embed_dim, dropout=0.1) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.dropout(x)
        # Cycle through best_arch if final depth > proxy depth.
        for i, cell in enumerate(self.cells):
            cell_arch = self.best_arch[i % len(self.best_arch)]
            x = cell(x, cell_arch)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

# ====== Evaluation & Final Training ======

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            _, pred = out.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0

def train_final_model(best_arch, device='cuda', lr=0.001, weight_decay=0.05,
                      epochs=650, batch_size=96):
    """
    Train the final discovered transformer network from scratch on CIFAR-10.
    Uses AdamW with cosine annealing and gradient clipping.
    """
    print("=== Training Final Discovered Architecture ===")
    train_loader, val_loader = get_cifar10_loaders(batch_size=batch_size, cutout_length=16)
    model = DiscoveredTransformerNetwork(img_size=32, patch_size=4, embed_dim=192,
                                          depth=20, num_classes=10, best_arch=best_arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs} - Training", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, pred = logits.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        scheduler.step()
        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        val_acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, val_acc)
        print(f"[Epoch {ep+1}/{epochs}] Train Loss={train_loss:.3f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}% (Best={best_acc:.2f}%)")
    print(f"=== Final Model Training Complete. Best Val Acc={best_acc:.2f}% ===")
    return best_acc

# ====== Main Function ======

def main():
    start_time = time.time()
    seed = 777
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Device: cuda")
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU")
    
    # Create the TransformerBanditNAS orchestrator (proxy network with 8 cells)
    transformer_nas = TransformerBanditNAS(
        device=device,
        alpha=1.0,
        lr=0.001,
        weight_decay=0.05,
        epochs=50,
        batch_size=96,
        depth=8
    )
    # Run the search phase with B=2500 child candidates per epoch.
    best_arch = transformer_nas.search(B=2500, warmup_epochs=5)
    print("\n[Main] Best Discovered Cell Configuration:")
    print(best_arch)
    # Train the final discovered architecture from scratch for 650 epochs on CIFAR-10.
    final_acc = train_final_model(best_arch=best_arch, device=device, epochs=650, batch_size=96)
    print(f"[Main] Final discovered architecture accuracy on CIFAR-10: {final_acc:.2f}%")
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total script runtime: {total_runtime:.2f} seconds")

if __name__ == "__main__":
    main()
