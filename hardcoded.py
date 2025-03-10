#!/usr/bin/env python3
"""
CMAB-NAS via full Nested Monte Carlo Search (NMCS) for CIFAR-10
"""

import os
import random
import math
import copy
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# ====== CIFAR-10 Data Loading ======

def get_cifar10_loaders(batch_size=96, num_workers=2, cutout_length=16):
    # Standard CIFAR-10 mean/std
    mean = (0.49139968, 0.48215827, 0.44653124)
    std  = (0.24703233, 0.24348505, 0.26158768)

    # Training transforms: random crop, horizontal flip, tensor conversion, normalization, Cutout augmentation
    transform_train_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if cutout_length > 0:
        transform_train_list.append(CutoutDefault(cutout_length))
    transform_train = transforms.Compose(transform_train_list)

    # Validation transforms: tensor conversion and normalization
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download CIFAR-10 datasets if needed
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    
    # On Windows, force num_workers to 0
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
        # img is a tensor of shape (C, H, W)
        h, w = img.shape[1], img.shape[2]
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img

# ====== Operations / Search Space ======

OPS = [
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'max_pool_3x3',
    'avg_pool_3x3',
]

def create_op(name, C_in, C_out, stride=1):
    """
    Create a PyTorch module implementing the given operation.
    """
    if name == 'skip_connect':
        if stride == 1:
            return nn.Identity()
        else:
            return FactorizedReduce(C_in, C_out, stride=stride)
    elif name == 'sep_conv_3x3':
        return SepConv(C_in, C_out, 3, stride, 1)
    elif name == 'sep_conv_5x5':
        return SepConv(C_in, C_out, 5, stride, 2)
    elif name == 'dil_conv_3x3':
        return DilConv(C_in, C_out, 3, stride, 2, dilation=2)
    elif name == 'dil_conv_5x5':
        return DilConv(C_in, C_out, 5, stride, 4, dilation=2)
    elif name == 'max_pool_3x3':
        return nn.MaxPool2d(3, stride=stride, padding=1)
    elif name == 'avg_pool_3x3':
        return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    else:
        raise ValueError(f"Unrecognized operation name: {name}")

class FactorizedReduce(nn.Module):
    """
    Reduces spatial dimensions (stride > 1) and aligns channels.
    Splits channels in half, applies two 1x1 convolutions (one on a shifted input),
    concatenates, then applies BatchNorm.
    """
    def __init__(self, C_in, C_out, stride=2):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0, "C_out must be even for FactorizedReduce"
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class DilConv(nn.Module):
    """
    Dilated depthwise separable convolution.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """
    Depthwise separable convolution repeated twice.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)

# ====== Proxy Network for Weight-Sharing: 8 Cells (Normal & Reduction) ======

class ProxyNetwork(nn.Module):
    """
    A weight-sharing network that stacks 8 cells.
    For reduction cells (indices 2 and 5), the cell will use a reduced
    input (via FactorizedReduce) so that spatial dimensions match the outputs
    of node operations.
    """
    def __init__(self, C_in=3, num_classes=10, num_cells=8, init_channels=16):
        super(ProxyNetwork, self).__init__()
        self._num_cells = num_cells
        self._init_channels = init_channels

        # Stem: initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels)
        )

        # Build cells; here, reduction cells at indices 2 and 5
        self.cells = nn.ModuleList()
        c_prev = init_channels
        for i in range(num_cells):
            reduction = (i in [2, 5])
            cell = MixedCell(c_prev, init_channels, reduction)
            self.cells.append(cell)
            c_prev = cell.out_channels  # Each cell outputs (init_channels * 4) channels

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, x, arch_normal=None, arch_reduce=None):
        """
        arch_normal, arch_reduce: lists describing the cell architecture for normal and reduction cells.
        Each is a list of 4 decisions (one per node in the cell). If None, a default skip cell is used.
        """
        s0 = self.stem(x)
        out = s0
        for cell in self.cells:
            if cell.reduction:
                out = cell(out, arch_reduce)
            else:
                out = cell(out, arch_normal)
        out = self.global_pooling(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

class MixedCell(nn.Module):
    """
    A cell composed of 4 nodes.
    Each node selects 2 inputs from:
       - -1: the external cell input (preprocessed)
       - 0..(node_index-1): previous node outputs

    To avoid spatial mismatches, if the cell is reduction, the external input is
    first reduced (via FactorizedReduce). Then all node operations run with stride=1.
    """
    def __init__(self, c_in, c_out, reduction=False):
        super(MixedCell, self).__init__()
        self.reduction = reduction
        # Final cell output: 4 * c_out channels (one per node)
        self.out_channels = c_out * 4

        if reduction:
            self.preprocess = FactorizedReduce(c_in, c_out, stride=2)
        else:
            self.preprocess = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, input_tensor, arch):
        """
        arch: list of 4 node definitions.
              Each definition is a tuple ((i1, op1), (i2, op2)).
              If None, a default skip cell is used.
        """
        if arch is None:
            arch = [ ((-1, 'skip_connect'), (-1, 'skip_connect')) ] * 4

        s0 = self.preprocess(input_tensor)
        node_outputs = []
        stride = 1

        def get_input(idx):
            return s0 if idx == -1 else node_outputs[idx]
        for node_def in arch:
            (i1, op1), (i2, op2) = node_def
            x1 = get_input(i1)
            x2 = get_input(i2)
            out1 = create_op(op1, x1.shape[1], s0.shape[1], stride=1).to(x1.device)(x1)
            out2 = create_op(op2, x2.shape[1], s0.shape[1], stride=1).to(x2.device)(x2)
            node_out = out1 + out2
            node_outputs.append(node_out)
        return torch.cat(node_outputs, dim=1)

# ====== NMCS Tree Search for a Single Cell ======

class NMCSNode:
    def __init__(self, level, node_index):
        """
        level: depth in the tree (0 for the root; a complete cell has level==num_nodes)
        node_index: the index of the cell node this decision corresponds to (0,1,2,3)
        """
        self.level = level
        self.node_index = node_index
        self.children = dict()  # mapping: arm (decision) -> NMCSNode
        self.visits = 0
        self.total_reward = 0.0
        self.parent = None
        self.arm_chosen = None  # the arm that led from parent to this node
        self.available_arms = self._build_arms_for_node(node_index)

    def _build_arms_for_node(self, node_index):
        valid_inputs = [-1] + list(range(node_index))
        arms = []
        for i1 in valid_inputs:
            for op1 in OPS:
                for i2 in valid_inputs:
                    for op2 in OPS:
                        if i1 == i2 and op1 == op2:
                            continue
                        if (i1 < i2) or (i1 == i2 and op1 < op2):
                            arms.append(((i1, op1), (i2, op2)))
        return arms

    def is_fully_expanded(self):
        return len(self.children) == len(self.available_arms)

    def average_reward(self):
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

class NMCSCellTree:
    def __init__(self, num_nodes=4, alpha=1.0):
        self.num_nodes = num_nodes  # depth of tree (cell has 4 nodes)
        self.alpha = alpha          # exploration parameter for UCB
        self.root = NMCSNode(level=0, node_index=0)

    def sample_architecture(self, use_exploration=True):
        node = self.root
        arch = []
        while node.level < self.num_nodes:
            if not node.is_fully_expanded() and use_exploration:
                unexplored = [arm for arm in node.available_arms if arm not in node.children]
                chosen_arm = random.choice(unexplored)
            else:
                best_val = -1e9
                chosen_arm = None
                for arm, child in node.children.items():
                    if child.visits == 0:
                        ucb = 1e6
                    else:
                        ucb = child.average_reward() + self.alpha * math.sqrt(2 * math.log(node.visits + 1) / child.visits)
                    if ucb > best_val:
                        best_val = ucb
                        chosen_arm = arm
                if chosen_arm is None:
                    chosen_arm = random.choice(node.available_arms)
            if chosen_arm not in node.children:
                child_node = NMCSNode(level=node.level + 1, node_index=node.level + 1)
                child_node.parent = node
                child_node.arm_chosen = chosen_arm
                node.children[chosen_arm] = child_node
            arch.append(chosen_arm)
            node = node.children[chosen_arm]
        return arch

    def rollout(self, partial_arch):
        arch = partial_arch.copy()
        current_level = len(arch)
        while current_level < self.num_nodes:
            dummy_node = NMCSNode(level=current_level, node_index=current_level)
            chosen_arm = random.choice(dummy_node.available_arms)
            arch.append(chosen_arm)
            current_level += 1
        return arch

    def backpropagate(self, arch, reward):
        node = self.root
        node.visits += 1
        node.total_reward += reward
        for arm in arch:
            if arm in node.children:
                node = node.children[arm]
                node.visits += 1
                node.total_reward += reward
            else:
                break

    def simulate(self, eval_fn, L=8):
        for _ in range(L):
            node = self.root
            arch = []
            while node.level < self.num_nodes:
                if not node.is_fully_expanded():
                    unexplored = [arm for arm in node.available_arms if arm not in node.children]
                    chosen_arm = random.choice(unexplored)
                    child_node = NMCSNode(level=node.level + 1, node_index=node.level + 1)
                    child_node.parent = node
                    child_node.arm_chosen = chosen_arm
                    node.children[chosen_arm] = child_node
                    arch.append(chosen_arm)
                    node = child_node
                    break
                else:
                    best_val = -1e9
                    chosen_arm = None
                    for arm, child in node.children.items():
                        if child.visits == 0:
                            ucb = 1e6
                        else:
                            ucb = child.average_reward() + self.alpha * math.sqrt(2 * math.log(node.visits + 1) / child.visits)
                        if ucb > best_val:
                            best_val = ucb
                            chosen_arm = arm
                    if chosen_arm is None:
                        chosen_arm = random.choice(node.available_arms)
                    arch.append(chosen_arm)
                    node = node.children[chosen_arm]
            if len(arch) < self.num_nodes:
                arch = self.rollout(arch)
            reward = eval_fn(arch)
            self.backpropagate(arch, reward)

    def best_architecture(self):
        arch = []
        node = self.root
        while node.level < self.num_nodes:
            best_avg = -1e9
            best_arm = None
            for arm, child in node.children.items():
                avg = child.average_reward()
                if avg > best_avg:
                    best_avg = avg
                    best_arm = arm
            if best_arm is None:
                best_arm = random.choice(node.available_arms)
            arch.append(best_arm)
            node = node.children[best_arm]
        return arch

# ====== NMCSNAS Orchestrator (Search) ======

class NMCSNAS:
    def __init__(self, device, alpha=1.0, lr=0.025, momentum=0.9,
                 weight_decay=3e-4, epochs=50, batch_size=96):
        self.device = device
        self.alpha = alpha
        self.epochs = epochs
        self.nmcs_tree_normal = NMCSCellTree(num_nodes=4, alpha=alpha)
        self.nmcs_tree_reduce = NMCSCellTree(num_nodes=4, alpha=alpha)
        self.proxy = ProxyNetwork(C_in=3, num_classes=10, num_cells=8, init_channels=16).to(device)
        self.optimizer = optim.SGD(self.proxy.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.val_loader = get_cifar10_loaders(batch_size=batch_size, cutout_length=16)
        self.best_arch_normal = None
        self.best_arch_reduce = None
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
                arch_n = self.nmcs_tree_normal.rollout([])
                arch_r = self.nmcs_tree_reduce.rollout([])
                self.optimizer.zero_grad()
                logits = self.proxy(inputs, arch_normal=arch_n, arch_reduce=arch_r)
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

    def _train_child_architecture(self, arch_n, arch_r, train_iter):
        self.proxy.train()
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(self.train_loader)
            inputs, targets = next(train_iter)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        logits = self.proxy(inputs, arch_normal=arch_n, arch_reduce=arch_r)
        loss = self.criterion(logits, targets)
        loss.backward()
        self.optimizer.step()
        return train_iter

    def _eval_child_architecture(self, arch_n, arch_r, num_batches=2):
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
                out = self.proxy(inputs, arch_normal=arch_n, arch_reduce=arch_r)
                _, pred = out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        return 100.0 * correct / total if total > 0 else 0.0

    def _eval_full_architecture(self, arch_n, arch_r):
        self.proxy.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                out = self.proxy(inputs, arch_normal=arch_n, arch_reduce=arch_r)
                _, pred = out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        return 100.0 * correct / total if total > 0 else 0.0

    def _evaluate_architecture_for_nmcs(self, arch, cell_type='normal'):
        self.proxy.eval()
        correct = 0
        total = 0
        val_iter = iter(self.val_loader)
        with torch.no_grad():
            for _ in range(1):
                try:
                    inputs, targets = next(val_iter)
                except StopIteration:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if cell_type == 'normal':
                    out = self.proxy(inputs, arch_normal=arch, arch_reduce=None)
                else:
                    out = self.proxy(inputs, arch_normal=None, arch_reduce=arch)
                _, pred = out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        return 100.0 * correct / total if total > 0 else 0.0

    def search(self, B=2500, warmup_epochs=5, simulation_iters=8):
        if not self.warmup_done:
            self._warmup_proxy(warmup_epochs)
        train_iter = iter(self.train_loader)
        for ep in range(self.epochs):
            self.current_epoch = ep
            print(f"\n=== Search Epoch {ep+1}/{self.epochs} ===")
            print(f"  (NMCS alpha: {self.alpha:.3f})")
            self.nmcs_tree_normal.simulate(lambda arch: self._evaluate_architecture_for_nmcs(arch, cell_type='normal'),
                                             L=simulation_iters)
            self.nmcs_tree_reduce.simulate(lambda arch: self._evaluate_architecture_for_nmcs(arch, cell_type='reduce'),
                                           L=simulation_iters)
            candidate_normal = []
            candidate_reduce = []
            for _ in range(B):
                arch_n = self.nmcs_tree_normal.sample_architecture(use_exploration=True)
                arch_r = self.nmcs_tree_reduce.sample_architecture(use_exploration=True)
                candidate_normal.append(arch_n)
                candidate_reduce.append(arch_r)
            for i in tqdm(range(B), desc=f"Epoch {ep+1}/{self.epochs} - Batch Progress", leave=False):
                arch_n = candidate_normal[i]
                arch_r = candidate_reduce[i]
                train_iter = self._train_child_architecture(arch_n, arch_r, train_iter)
                reward = self._eval_child_architecture(arch_n, arch_r, num_batches=2)
                self.nmcs_tree_normal.backpropagate(arch_n, reward)
                self.nmcs_tree_reduce.backpropagate(arch_r, reward)
            self.alpha *= 0.95
            self.nmcs_tree_normal.alpha = self.alpha
            self.nmcs_tree_reduce.alpha = self.alpha
            self.nmcs_tree_normal.simulate(lambda arch: self._evaluate_architecture_for_nmcs(arch, cell_type='normal'), L=800)
            self.nmcs_tree_reduce.simulate(lambda arch: self._evaluate_architecture_for_nmcs(arch, cell_type='reduce'), L=800)
            arch_n_opt = self.nmcs_tree_normal.best_architecture()
            arch_r_opt = self.nmcs_tree_reduce.best_architecture()
            acc_opt = self._eval_full_architecture(arch_n_opt, arch_r_opt)
            print(f"  Local-opt architecture validation accuracy: {acc_opt:.2f}%")
            if acc_opt > self.best_acc:
                self.best_acc = acc_opt
                self.best_arch_normal = copy.deepcopy(arch_n_opt)
                self.best_arch_reduce = copy.deepcopy(arch_r_opt)
                print(f"  [Update] New best architecture found: {acc_opt:.2f}%")
        print(f"\n=== SEARCH FINISHED ===")
        print(f"Best discovered architecture validation accuracy: {self.best_acc:.2f}%")
        return self.best_arch_normal, self.best_arch_reduce

# ====== Final Discovered Architecture (Stand-Alone) ======

# Best architecture as given in the test description:
NORMAL_CELL_ARCH = [
    ((-2, 'sep_conv_3x3'), (-1, 'sep_conv_5x5')),
    ((-2, 'dil_conv_5x5'), (-2, 'skip_connect')),
    ((0, 'sep_conv_5x5'), (1, 'dil_conv_5x5')),
    ((0, 'sep_conv_5x5'), (1, 'dil_conv_5x5'))
]

REDUCTION_CELL_ARCH = [
    ((-1, 'dil_conv_5x5'), (-1, 'skip_connect')),
    ((-2, 'dil_conv_3x3'), (-2, 'skip_connect')),
    ((-2, 'max_pool_3x3'), (0, 'skip_connect')),
    ((0, 'skip_connect'), (1, 'sep_conv_3x3'))
]

class FinalCell(nn.Module):
    """
    Final cell for the stand-alone network.
    If reduction is True, the external inputs are reduced.
    This cell accepts two external inputs:
      - index -2 corresponds to c_{k-2}
      - index -1 corresponds to c_{k-1}
    """
    def __init__(self, c_in0, c_in1, c_out, arch, reduction=False):
        super(FinalCell, self).__init__()
        self.reduction = reduction
        self.out_channels = c_out * 4
        if reduction:
            self.preprocess0 = FactorizedReduce(c_in0, c_out, stride=2)
            self.preprocess1 = FactorizedReduce(c_in1, c_out, stride=2)
        else:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(c_in0, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.preprocess1 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(c_in1, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )
        self.arch = arch

    def forward(self, inputs):
        # inputs is a tuple: (c_{k-2}, c_{k-1})
        s0 = self.preprocess0(inputs[0])
        s1 = self.preprocess1(inputs[1])
        node_outputs = []
        def get_input(idx):
            if idx == -2:
                ret = s0
            elif idx == -1:
                ret = s1
            else:
                ret = node_outputs[idx]
            # Ensure spatial sizes match s0 (the chosen baseline)
            if ret.shape[2] != s0.shape[2] or ret.shape[3] != s0.shape[3]:
                ret = nn.functional.interpolate(ret, size=(s0.shape[2], s0.shape[3]), mode='bilinear', align_corners=False)
            return ret
        for (i1, op1), (i2, op2) in self.arch:
            x1 = get_input(i1)
            x2 = get_input(i2)
            out1 = create_op(op1, x1.shape[1], s0.shape[1], stride=1).to(x1.device)(x1)
            out2 = create_op(op2, x2.shape[1], s0.shape[1], stride=1).to(x2.device)(x2)
            node_out = out1 + out2
            node_outputs.append(node_out)
        return torch.cat(node_outputs, dim=1)

class DiscoveredNetwork(nn.Module):
    """
    Stand-alone network built from discovered cells.
    It stacks 20 cells (with reduction cells at roughly 1/3 and 2/3 of the depth).
    This network accepts two stem outputs as initial inputs.
    """
    def __init__(self, C_in=3, num_classes=10, init_channels=36, num_cells=20,
                 arch_normal=NORMAL_CELL_ARCH, arch_reduce=REDUCTION_CELL_ARCH):
        super(DiscoveredNetwork, self).__init__()
        self.num_cells = num_cells
        self.arch_normal = arch_normal
        self.arch_reduce = arch_reduce
        self.stem0 = nn.Sequential(
            nn.Conv2d(C_in, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels)
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(init_channels, init_channels, 1, bias=False),
            nn.BatchNorm2d(init_channels)
        )
        reduction_positions = [num_cells // 3, (2 * num_cells) // 3]
        self.cells = nn.ModuleList()
        c_prev0, c_prev1 = init_channels, init_channels
        for i in range(num_cells):
            reduction = (i in reduction_positions)
            cell = FinalCell(c_prev0, c_prev1, init_channels, arch_reduce if reduction else arch_normal, reduction)
            self.cells.append(cell)
            c_prev0, c_prev1 = c_prev1, init_channels * 4
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev1, num_classes)

    def forward(self, x):
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        for cell in self.cells:
            s0, s1 = s1, cell((s0, s1))
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

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

def train_final_model(arch_normal, arch_reduce, device='cuda',
                      lr=0.025, momentum=0.9, weight_decay=3e-4,
                      epochs=650, batch_size=96):
    """
    Train the discovered architecture from scratch on CIFAR-10.
    """
    print("=== Training Final Discovered Architecture ===")
    train_loader, val_loader = get_cifar10_loaders(batch_size=batch_size, cutout_length=16)
    model = DiscoveredNetwork(C_in=3, num_classes=10, init_channels=36, num_cells=20,
                              arch_normal=arch_normal, arch_reduce=arch_reduce).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
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
    start_time = time.time()  # Start the timer
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
    print("Using the following best architectures for final training:")
    print("Normal Cell:", NORMAL_CELL_ARCH)
    print("Reduction Cell:", REDUCTION_CELL_ARCH)
    final_acc = train_final_model(
        arch_normal=NORMAL_CELL_ARCH,
        arch_reduce=REDUCTION_CELL_ARCH,
        device=device,
        epochs=650,
        batch_size=96
    )
    print(f"[Main] Final discovered architecture accuracy on CIFAR-10: {final_acc:.2f}%")
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total script runtime: {total_runtime:.2f} seconds")

# TODO: fix this later, training loop is cooked
if __name__ == "__main__":
    main()