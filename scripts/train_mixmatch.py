#!/usr/bin/env python3
"""
Minimal DivideMix-ish trainer in plain PyTorch (single file).

Assumes ImageFolder directories:
  train_dir/class_x/xxx.png
  val_dir/class_x/xxx.png
  test_dir/class_x/xxx.png  (optional)

Core features:
- Warmup: CE + confidence penalty (CE - entropy)
- Co-divide each epoch via GMM on per-sample losses
- Two-stream sampling: labeled + unlabeled (based on other net's clean probs)
- MixMatch-style: pseudo-labeling (eval-mode), sharpening, MixUp, unsup MSE
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_rampup(current: float, rampup_length: int) -> float:
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, float(rampup_length)))
    return current / float(rampup_length)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # targets: [B, C] probabilities
    logp = F.log_softmax(logits, dim=1)
    return -(targets * logp).sum(dim=1).mean()


def sharpen(probs: torch.Tensor, T: float) -> torch.Tensor:
    if T <= 0:
        return probs
    p = probs ** (1.0 / T)
    return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)


def confidence_penalty(logits: torch.Tensor) -> torch.Tensor:
    """
    Confidence penalty = sum p log p = -Entropy(p). Negative number.
    Adding this to CE => CE - Entropy, encourages higher entropy (less overconfidence).
    """
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    return -(p * logp).sum(dim=1).mean()


def normal_pdf(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    var = var.clamp_min(1e-6)
    return torch.exp(-0.5 * (x - mean) ** 2 / var) / torch.sqrt(2.0 * torch.pi * var)


@torch.no_grad()
def gmm_clean_posterior(losses: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """
    Fit 2-component 1D GMM on normalized losses and return posterior prob of "clean"
    (the component with smaller mean loss).

    losses: [N] on device
    returns: [N] in [0,1]
    """
    eps = 1e-6
    x = (losses - losses.min()) / (losses.max() - losses.min() + eps)
    x = x.clamp(0.0, 1.0)

    mu1 = torch.quantile(x, 0.3)
    mu2 = torch.quantile(x, 0.7)
    var1 = x.var(unbiased=False) + eps
    var2 = x.var(unbiased=False) + eps
    pi1 = torch.tensor(0.5, device=x.device)
    pi2 = torch.tensor(0.5, device=x.device)

    for _ in range(iters):
        p1 = pi1 * normal_pdf(x, mu1, var1)
        p2 = pi2 * normal_pdf(x, mu2, var2)
        denom = (p1 + p2).clamp_min(eps)
        gamma1 = p1 / denom
        gamma2 = p2 / denom

        n1 = gamma1.sum().clamp_min(eps)
        n2 = gamma2.sum().clamp_min(eps)

        mu1 = (gamma1 * x).sum() / n1
        mu2 = (gamma2 * x).sum() / n2
        var1 = (gamma1 * (x - mu1) ** 2).sum() / n1 + eps
        var2 = (gamma2 * (x - mu2) ** 2).sum() / n2 + eps
        pi1 = n1 / x.numel()
        pi2 = n2 / x.numel()

    # recompute with final params
    p1 = pi1 * normal_pdf(x, mu1, var1)
    p2 = pi2 * normal_pdf(x, mu2, var2)
    denom = (p1 + p2).clamp_min(eps)
    gamma1 = p1 / denom
    gamma2 = p2 / denom

    clean = gamma1 if (mu1 <= mu2) else gamma2
    return clean.clamp(0.0, 1.0)


def macro_f1_from_confmat(conf: torch.Tensor) -> float:
    """
    conf: [C,C] where rows=true, cols=pred
    """
    conf = conf.float()
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = (2 * tp + fp + fn).clamp_min(1e-12)
    f1 = (2 * tp) / denom
    return f1.mean().item()


# -------------------------
# Datasets
# -------------------------

class TwoViewImageFolder(Dataset):
    def __init__(self, root: str, transform1, transform2):
        self.ds = datasets.ImageFolder(root=root)
        self.t1 = transform1
        self.t2 = transform2

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x, y = self.ds[idx]
        return self.t1(x), self.t2(x), y, idx


class EvalImageFolder(Dataset):
    def __init__(self, root: str, transform):
        self.ds = datasets.ImageFolder(root=root)
        self.t = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x, y = self.ds[idx]
        return self.t(x), y, idx


# -------------------------
# Model
# -------------------------

def build_mobilenet(num_classes: int, pretrained: bool = True) -> nn.Module:
    m = mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
    # Replace classifier head
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


# -------------------------
# DivideMix core step
# -------------------------

@torch.no_grad()
def estimate_clean_probs(
    net: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    num_train: int,
    gmm_iters: int,
) -> torch.Tensor:
    """
    Compute per-sample CE loss (no aug) and fit GMM -> clean posterior.
    Returns: clean_prob [N] on CPU (float32)
    """
    net.eval()
    losses = torch.empty(num_train, device=device)
    seen = torch.zeros(num_train, device=device, dtype=torch.bool)

    for x, y, idx in eval_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True).long()

        logits = net(x)
        l = F.cross_entropy(logits, y, reduction="none")
        losses[idx] = l
        seen[idx] = True

    if not torch.all(seen):
        missing = int((~seen).sum().item())
        raise RuntimeError(
            f"eval_loader missed {missing} samples. Ensure shuffle=False, drop_last=False, and global indices."
        )

    prob = gmm_clean_posterior(losses, iters=gmm_iters)
    return prob.detach().float().cpu()


def make_split_subsets(
    full_dataset: TwoViewImageFolder,
    clean_prob_other: torch.Tensor,  # [N] on CPU
    p_threshold: float,
) -> Tuple[Subset, Subset]:
    """
    Based on other net's clean probs, split dataset into labeled/unlabeled Subsets.
    """
    clean_prob_other = clean_prob_other.clamp(0.0, 1.0)
    idx_clean = torch.nonzero(clean_prob_other >= p_threshold, as_tuple=False).squeeze(1).tolist()
    idx_noisy = torch.nonzero(clean_prob_other < p_threshold, as_tuple=False).squeeze(1).tolist()
    labeled = Subset(full_dataset, idx_clean)
    unlabeled = Subset(full_dataset, idx_noisy)
    return labeled, unlabeled


@torch.no_grad()
def get_pseudo_targets(
    net_a: nn.Module,
    net_b: nn.Module,
    x_u1: torch.Tensor,
    x_u2: torch.Tensor,
    T: float,
) -> torch.Tensor:
    """
    Pseudo label for unlabeled: avg predictions from both nets on both views, then sharpen.
    BN-safe: caller sets eval mode.
    """
    pu = (
        F.softmax(net_a(x_u1), dim=1) +
        F.softmax(net_a(x_u2), dim=1) +
        F.softmax(net_b(x_u1), dim=1) +
        F.softmax(net_b(x_u2), dim=1)
    ) / 4.0
    return sharpen(pu, T)


def mixmatch_step(
    net: nn.Module,
    other_net: nn.Module,
    batch_l: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_u: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    clean_prob_for_l: Optional[torch.Tensor],  # per-sample weights for labeled batch, CPU tensor indexed by idx
    num_classes: int,
    T: float,
    alpha: float,
    w_u: float,
    lambda_p: float,
    device: torch.device,
    use_empirical_prior: bool = False,
) -> torch.Tensor:
    """
    One optimization step loss for a single net (net) using labeled+unlabeled batches.
    """
    x_l1, x_l2, y_l, idx_l = batch_l
    x_u1, x_u2, _, _ = batch_u

    x_l1 = x_l1.to(device, non_blocking=True)
    x_l2 = x_l2.to(device, non_blocking=True)
    y_l = y_l.to(device, non_blocking=True)
    idx_l = idx_l.to(device, non_blocking=True).long()

    x_u1 = x_u1.to(device, non_blocking=True)
    x_u2 = x_u2.to(device, non_blocking=True)

    y_l_oh = F.one_hot(y_l, num_classes=num_classes).float()

    # Per-sample "clean weight" for labeled refinement
    if clean_prob_for_l is not None:
        w_x = clean_prob_for_l[idx_l.cpu()].to(device).unsqueeze(1).clamp(0.0, 1.0)
    else:
        w_x = torch.ones((x_l1.size(0), 1), device=device)

    # Pseudo targets + refined labeled targets (BN-safe eval)
    net_was_train = net.training
    other_was_train = other_net.training
    net.eval()
    other_net.eval()
    with torch.no_grad():
        # unlabeled pseudo
        targets_u = get_pseudo_targets(net, other_net, x_u1, x_u2, T=T)
        # labeled refinement: interpolate between GT and model prediction
        px = (F.softmax(net(x_l1), dim=1) + F.softmax(net(x_l2), dim=1)) / 2.0
        px = w_x * y_l_oh + (1.0 - w_x) * px
        targets_x = sharpen(px, T=T)
    net.train(net_was_train)
    other_net.train(other_was_train)

    # MixUp over [x_l1, x_l2, x_u1, x_u2]
    inputs = torch.cat([x_l1, x_l2, x_u1, x_u2], dim=0)
    targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

    if inputs.size(0) == 0:
        return torch.tensor(0.0, device=device)

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)

    perm = torch.randperm(inputs.size(0), device=device)
    inputs_b = inputs[perm]
    targets_b = targets[perm]

    mixed_x = lam * inputs + (1.0 - lam) * inputs_b
    mixed_y = lam * targets + (1.0 - lam) * targets_b

    logits = net(mixed_x)

    n_l = x_l1.size(0)
    # supervised portion = first 2*n_l
    if n_l > 0:
        Lx = soft_cross_entropy(logits[: 2 * n_l], mixed_y[: 2 * n_l])
    else:
        Lx = torch.tensor(0.0, device=device)

    # unsupervised portion = rest
    if logits.size(0) > 2 * n_l:
        probs_u = F.softmax(logits[2 * n_l :], dim=1)
        Lu = torch.mean((probs_u - mixed_y[2 * n_l :]) ** 2)
    else:
        Lu = torch.tensor(0.0, device=device)

    # prior penalty to prevent collapse
    if use_empirical_prior and n_l > 0:
        # empirical prior from labeled y in this batch
        prior = y_l_oh.mean(dim=0).clamp_min(1e-6)
        prior = prior / prior.sum()
    else:
        prior = torch.full((num_classes,), 1.0 / num_classes, device=device)

    pred_mean = F.softmax(logits, dim=1).mean(dim=0)
    penalty = torch.sum(prior * torch.log(prior / pred_mean.clamp_min(1e-12)))

    return Lx + (w_u * Lu) + (lambda_p * penalty)


# -------------------------
# Train / Eval
# -------------------------

@torch.no_grad()
def evaluate(net1: nn.Module, net2: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    net1.eval()
    net2.eval()
    conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = 0.5 * (net1(x) + net2(x))
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)

        total += y.numel()
        correct += (preds == y).sum().item()
        loss_sum += loss.item() * y.size(0)

        for t, p in zip(y, preds):
            conf[t.long(), p.long()] += 1

    acc = correct / max(1, total)
    f1_macro = macro_f1_from_confmat(conf)
    loss_mean = loss_sum / max(1, total)
    return loss_mean, acc, f1_macro


def cycle(loader: DataLoader) -> Iterable:
    while True:
        for batch in loader:
            yield batch


@dataclass
class Config:
    train_dir: str
    val_dir: str
    test_dir: Optional[str]
    img_size: int
    batch_size: int
    num_workers: int
    epochs: int
    warmup_epochs: int
    lr: float
    weight_decay: float
    momentum: float
    p_threshold: float
    lambda_u: float
    lambda_p: float
    lambda_cp: float
    temperature: float
    mixup_alpha: float
    rampup_length: int
    gmm_iters: int
    pretrained: bool
    amp: bool
    seed: int
    use_empirical_prior: bool
    device: int


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    
    # Create unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "train.log")
    
    def log(msg: str) -> None:
        """Print to console and write to log file."""
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    
    log(f"Device: {device}")
    log(f"Run directory: {run_dir}")

    # Transforms (simple, tweak as needed)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_t1 = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_t2 = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_t = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = TwoViewImageFolder(cfg.train_dir, train_t1, train_t2)
    eval_train_ds = EvalImageFolder(cfg.train_dir, eval_t)
    val_ds = datasets.ImageFolder(cfg.val_dir, transform=eval_t)

    num_classes = len(train_ds.ds.classes)
    log(f"Classes: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_eval_loader = DataLoader(
        eval_train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    net1 = build_mobilenet(num_classes=num_classes, pretrained=cfg.pretrained).to(device)
    net2 = build_mobilenet(num_classes=num_classes, pretrained=cfg.pretrained).to(device)

    opt1 = torch.optim.SGD(net1.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    opt2 = torch.optim.SGD(net2.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # simple cosine schedule (you can swap)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.epochs)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    best_f1 = -1.0

    # Initialize clean probs (all ones) on CPU
    clean_prob1 = torch.ones(len(train_ds), dtype=torch.float32)
    clean_prob2 = torch.ones(len(train_ds), dtype=torch.float32)

    for epoch in range(cfg.epochs):
        net1.train()
        net2.train()

        # --- Co-divide (after warmup) ---
        if epoch >= cfg.warmup_epochs:
            clean_prob1 = estimate_clean_probs(net1, train_eval_loader, device, len(train_ds), cfg.gmm_iters)
            clean_prob2 = estimate_clean_probs(net2, train_eval_loader, device, len(train_ds), cfg.gmm_iters)

            clean_ratio1 = clean_prob1.mean().item()
            clean_ratio2 = clean_prob2.mean().item()
            log(f"[epoch {epoch}] clean_ratio net1={clean_ratio1:.3f} net2={clean_ratio2:.3f}")

            # Split sets for each net based on OTHER net
            labeled1, unlabeled1 = make_split_subsets(train_ds, clean_prob2, cfg.p_threshold)
            labeled2, unlabeled2 = make_split_subsets(train_ds, clean_prob1, cfg.p_threshold)

            # Two-stream loaders (IMPORTANT difference vs splitting within-batch)
            loader_l1 = DataLoader(labeled1, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                   num_workers=cfg.num_workers, pin_memory=True)
            loader_u1 = DataLoader(unlabeled1, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                   num_workers=cfg.num_workers, pin_memory=True)

            loader_l2 = DataLoader(labeled2, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                   num_workers=cfg.num_workers, pin_memory=True)
            loader_u2 = DataLoader(unlabeled2, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                   num_workers=cfg.num_workers, pin_memory=True)

            it_l1 = cycle(loader_l1)
            it_u1 = cycle(loader_u1) if len(unlabeled1) > 0 else None
            it_l2 = cycle(loader_l2)
            it_u2 = cycle(loader_u2) if len(unlabeled2) > 0 else None

            # Choose steps per epoch based on labeled stream size (simple)
            steps = max(1, len(loader_l1))
        else:
            # Warmup uses standard train loader
            warm_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                     num_workers=cfg.num_workers, pin_memory=True)
            steps = len(warm_loader)

        running_loss = 0.0

        for step in range(steps):
            # Ramp unsup weight
            progress = (epoch - cfg.warmup_epochs) + (step / max(1, steps))
            w_u = cfg.lambda_u * linear_rampup(progress, cfg.rampup_length) if epoch >= cfg.warmup_epochs else 0.0

            if epoch < cfg.warmup_epochs:
                for step, (x1, x2, y, _) in enumerate(warm_loader):
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    # net1 warmup
                    opt1.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                        logits1a = net1(x1)
                        logits1b = net1(x2)
                        ce1 = 0.5 * (F.cross_entropy(logits1a, y) + F.cross_entropy(logits1b, y))
                        cp1 = 0.5 * (confidence_penalty(logits1a) + confidence_penalty(logits1b))
                        loss1 = ce1 - cfg.lambda_cp * cp1
                    scaler.scale(loss1).backward()
                    scaler.step(opt1)

                    # net2 warmup
                    opt2.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                        logits2a = net2(x1)
                        logits2b = net2(x2)
                        ce2 = 0.5 * (F.cross_entropy(logits2a, y) + F.cross_entropy(logits2b, y))
                        cp2 = 0.5 * (confidence_penalty(logits2a) + confidence_penalty(logits2b))
                        loss2 = ce2 - cfg.lambda_cp * cp2
                    scaler.scale(loss2).backward()
                    scaler.step(opt2)

                    scaler.update()

                    loss = (loss1 + loss2).detach()
                    running_loss += loss.item()
                continue

            # --- DivideMix training epoch ---
            batch_l1 = next(it_l1)
            batch_u1 = next(it_u1) if it_u1 is not None else batch_l1  # fallback if no unlabeled
            batch_l2 = next(it_l2)
            batch_u2 = next(it_u2) if it_u2 is not None else batch_l2

            # Update net1
            opt1.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss1 = mixmatch_step(
                    net=net1,
                    other_net=net2,
                    batch_l=batch_l1,
                    batch_u=batch_u1,
                    clean_prob_for_l=clean_prob2,  # weights from other net
                    num_classes=num_classes,
                    T=cfg.temperature,
                    alpha=cfg.mixup_alpha,
                    w_u=w_u,
                    lambda_p=cfg.lambda_p,
                    device=device,
                    use_empirical_prior=cfg.use_empirical_prior,
                )
            scaler.scale(loss1).backward()
            scaler.step(opt1)

            # Update net2
            opt2.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss2 = mixmatch_step(
                    net=net2,
                    other_net=net1,
                    batch_l=batch_l2,
                    batch_u=batch_u2,
                    clean_prob_for_l=clean_prob1,
                    num_classes=num_classes,
                    T=cfg.temperature,
                    alpha=cfg.mixup_alpha,
                    w_u=w_u,
                    lambda_p=cfg.lambda_p,
                    device=device,
                    use_empirical_prior=cfg.use_empirical_prior,
                )
            scaler.scale(loss2).backward()
            scaler.step(opt2)
            scaler.update()

            running_loss += (loss1.detach().item() + loss2.detach().item())

        sch1.step()
        sch2.step()

        val_loss, val_acc, val_f1 = evaluate(net1, net2, val_loader, device, num_classes)
        avg_loss = running_loss / max(1, steps)

        log(
            f"Epoch {epoch:03d} | train_loss {avg_loss:.4f} | val_loss {val_loss:.4f} "
            f"| val_acc {val_acc:.4f} | val_f1_macro {val_f1:.4f} | lr {sch1.get_last_lr()[0]:.6f}"
        )

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = os.path.join(run_dir, "best_dividemix.pt")
            torch.save(
                {"net1": net1.state_dict(), "net2": net2.state_dict(), "epoch": epoch, "val_f1": val_f1},
                checkpoint_path,
            )

    log(f"Best val macro-F1: {best_f1:.4f} (checkpoint: {os.path.join(run_dir, 'best_dividemix.pt')})")


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--val_dir", type=str, required=True)
    ap.add_argument("--test_dir", type=str, default=None)
    ap.add_argument("--img_size", type=int, default=480)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)  # closer to paper than 1e-5
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--p_threshold", type=float, default=0.8)
    ap.add_argument("--lambda_u", type=float, default=10.0)
    ap.add_argument("--lambda_p", type=float, default=0.1)
    ap.add_argument("--lambda_cp", type=float, default=1.0)      # confidence penalty weight (tune 0.1–1.0)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--mixup_alpha", type=float, default=4.0)
    ap.add_argument("--rampup_length", type=int, default=16)
    ap.add_argument("--gmm_iters", type=int, default=10)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_empirical_prior", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    return Config(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        p_threshold=args.p_threshold,
        lambda_u=args.lambda_u,
        lambda_p=args.lambda_p,
        lambda_cp=args.lambda_cp,
        temperature=args.temperature,
        mixup_alpha=args.mixup_alpha,
        rampup_length=args.rampup_length,
        gmm_iters=args.gmm_iters,
        pretrained=args.pretrained,
        amp=args.amp,
        seed=args.seed,
        use_empirical_prior=args.use_empirical_prior,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
