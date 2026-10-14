from __future__ import annotations


import random


import numpy as np


import torch


from torch import nn


from torch.utils.data import DataLoader, TensorDataset


from dataset1_dataset2_temporal_transport import REF_TIMES, SHARED_CHANNELS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_from_train(train_values: np.ndarray, train_masks: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros(len(SHARED_CHANNELS), dtype=np.float32)
    stds = np.ones(len(SHARED_CHANNELS), dtype=np.float32)
    for j in range(len(SHARED_CHANNELS)):
        observed = train_values[:, :, j][train_masks[:, :, j] > 0]
        if observed.size:
            means[j] = float(np.mean(observed))
            std = float(np.std(observed))
            stds[j] = std if std > 1e-6 else 1.0
    norm = (values - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)
    norm = np.where(np.isfinite(norm), norm, np.nan)
    return norm.astype(np.float32), means, stds


def prepare_inputs(train_values: np.ndarray, train_masks: np.ndarray, values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    norm_values, _, _ = normalize_from_train(train_values, train_masks, values)
    filled = np.nan_to_num(norm_values, nan=0.0)
    return np.concatenate([filled, masks], axis=2).astype(np.float32)


class PhenotypeTransformer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, len(REF_TIMES), d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(SHARED_CHANNELS)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)
        recon = self.decoder(h)
        logits = self.classifier(pooled)
        return pooled, recon, logits


def make_augmented(batch_x: torch.Tensor, channel_dim: int, min_keep: float = 0.3, max_keep: float = 0.7) -> torch.Tensor:
    values = batch_x[:, :, :channel_dim]
    masks = batch_x[:, :, channel_dim:]
    keep_prob = random.uniform(min_keep, max_keep)
    drop = torch.bernoulli(torch.full_like(masks, keep_prob))
    aug_masks = masks * drop
    aug_values = values * aug_masks
    return torch.cat([aug_values, aug_masks], dim=-1)


def train_model(
    train_values: np.ndarray,
    train_masks: np.ndarray,
    train_labels: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    alpha: float,
    beta: float,
    device: torch.device,
) -> tuple[PhenotypeTransformer, list[dict[str, float]]]:
    x_input = prepare_inputs(train_values, train_masks, train_values, train_masks)
    filled = x_input[:, :, : len(SHARED_CHANNELS)]
    dataset = TensorDataset(
        torch.from_numpy(x_input),
        torch.from_numpy(filled),
        torch.from_numpy(train_masks.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_classes = int(np.max(train_labels[train_labels >= 0]) + 1)
    class_counts = np.bincount(train_labels[train_labels >= 0], minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    model = PhenotypeTransformer(input_dim=x_input.shape[2], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        recon_total = 0.0
        cls_total = 0.0
        cons_total = 0.0
        acc_total = 0.0
        acc_batches = 0
        n_batches = 0
        for batch_x, batch_target, batch_mask, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            z, recon, logits = model(batch_x)
            aug_x = make_augmented(batch_x, channel_dim=len(SHARED_CHANNELS))
            z_aug, recon_aug, logits_aug = model(aug_x)

            recon_loss = (((recon - batch_target) * batch_mask).pow(2).sum() / torch.clamp(batch_mask.sum(), min=1.0))
            recon_aug_loss = (((recon_aug - batch_target) * batch_mask).pow(2).sum() / torch.clamp(batch_mask.sum(), min=1.0))
            label_mask = batch_y >= 0
            if torch.any(label_mask):
                cls_loss = 0.5 * (ce(logits[label_mask], batch_y[label_mask]) + ce(logits_aug[label_mask], batch_y[label_mask]))
                preds = torch.argmax(logits[label_mask], dim=1)
                acc_total += float((preds == batch_y[label_mask]).float().mean().item())
                acc_batches += 1
            else:
                cls_loss = torch.tensor(0.0, device=device)
            cos = nn.functional.cosine_similarity(z, z_aug, dim=1)
            cons_loss = (1.0 - cos).mean()
            loss = recon_loss + recon_aug_loss + alpha * cls_loss + beta * cons_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_total += float((recon_loss + recon_aug_loss).item())
            cls_total += float(cls_loss.item())
            cons_total += float(cons_loss.item())
            n_batches += 1
        history.append(
            {
                "epoch": epoch,
                "recon_loss": recon_total / max(n_batches, 1),
                "cls_loss": cls_total / max(n_batches, 1),
                "cons_loss": cons_total / max(n_batches, 1),
                "train_acc": acc_total / max(acc_batches, 1),
            }
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[train] epoch={epoch} recon={history[-1]['recon_loss']:.6f} cls={history[-1]['cls_loss']:.6f} cons={history[-1]['cons_loss']:.6f} acc={history[-1]['train_acc']:.4f}",
                flush=True,
            )
    return model, history


def predict(
    model: PhenotypeTransformer,
    train_values: np.ndarray,
    train_masks: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_input = prepare_inputs(train_values, train_masks, values, masks)
    loader = DataLoader(torch.from_numpy(x_input), batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    probs = []
    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            _, _, logits = model(batch_x)
            prob = torch.softmax(logits, dim=1)
            preds.append(torch.argmax(prob, dim=1).cpu().numpy())
            probs.append(prob.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(probs, axis=0)
