"""
incremental.py — HWD
File baru — tidak ada di FGTI maupun CSDI.

Tujuan:
  Fine-tuning model yang sudah terlatih menggunakan data baru
  (late response / periode baru IMK) tanpa full retrain dari awal.

  Strategi: frozen layers — hanya detail level paling halus
  dan output layers yang di-update. Layer global (approx, DWT,
  cross-attention) dibekukan karena pola tren makro stabil.

Cara pakai:
  # Dari train.py (sudah terintegrasi via --incremental flag)
  python train.py --incremental --base_ckpt checkpoints/hwd_imk.pt \
                  --dataset imk --data_path data/imk_2026.csv \
                  --incr_epochs 50 --incr_lr 1e-5

  # Atau dari notebook:
  import incremental, train, main_model
  model = main_model.HWD(configs).to(configs.device)
  model.load_state_dict(torch.load('checkpoints/hwd_imk.pt'))
  new_loader, _ = dataset.get_imk_dataset(configs)
  model = incremental.incremental_finetune(model, new_loader,
              freeze_ratio=0.8, lr=1e-5, epochs=50)
"""

import time
import numpy as np
import torch
from torch import optim


def _get_freeze_boundary(model, freeze_ratio: float) -> int:
    """
    Hitung index parameter terakhir yang di-freeze.
    Parameter diurutkan sesuai urutan model.named_parameters():
    layer awal (DWT, approx, cross-attn) → layer akhir (detail L1, output).

    Args:
        model        : HWD model instance
        freeze_ratio : proporsi parameter yang dibekukan (0.0–1.0)

    Returns:
        n_freeze : jumlah parameter (bukan tensor) yang di-freeze
    """
    all_params = list(model.named_parameters())
    n_freeze   = int(len(all_params) * freeze_ratio)
    return n_freeze


def freeze_layers(model, freeze_ratio: float = 0.8):
    """
    Bekukan freeze_ratio proporsi awal dari parameter model.
    Layer yang tidak dibekukan (trainable):
      - detail_encoders[-1]  : detail level paling halus (L1)
      - output_projection*   : output layers diffmodel
    """
    all_params = list(model.named_parameters())
    n_freeze   = _get_freeze_boundary(model, freeze_ratio)

    frozen_count    = 0
    trainable_count = 0

    for i, (name, param) in enumerate(all_params):
        if i < n_freeze:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()

    print(f"Frozen    : {frozen_count:,} params  ({freeze_ratio*100:.0f}%)")
    print(f"Trainable : {trainable_count:,} params  ({(1-freeze_ratio)*100:.0f}%)")
    return model


def unfreeze_all(model):
    """Kembalikan semua parameter ke trainable setelah fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def incremental_finetune(
    model,
    new_dataloader,
    freeze_ratio: float = 0.8,
    lr: float = 1e-5,
    epochs: int = 50,
    device: str = None,
):
    """
    Fine-tune model pada data baru dengan frozen layers.

    Args:
        model          : HWD instance yang sudah di-load checkpoint
        new_dataloader : DataLoader berisi data periode baru
        freeze_ratio   : proporsi parameter yang dibekukan (default 0.8)
        lr             : learning rate fine-tuning (jauh lebih kecil dari training awal)
        epochs         : jumlah epoch fine-tuning (default 50)
        device         : cuda/cpu — kalau None ambil dari model

    Returns:
        model yang sudah di-fine-tune
    """
    if device is None:
        device = next(model.parameters()).device

    print(f"\n{'─'*50}")
    print(f"Incremental fine-tuning")
    print(f"  freeze_ratio : {freeze_ratio}")
    print(f"  lr           : {lr}")
    print(f"  epochs       : {epochs}")
    print(f"  batches/epoch: {len(new_dataloader)}")
    print(f"{'─'*50}")

    # 1. Bekukan sebagian besar parameter
    model = freeze_layers(model, freeze_ratio)

    # 2. Optimizer hanya untuk parameter trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer        = optim.Adam(trainable_params, lr=lr, weight_decay=1e-6)

    # 3. Fine-tuning loop — pakai training loop yang sama dengan train.py
    model.train()
    for epoch in range(epochs):
        epoch_loss = []
        epoch_time = time.time()

        for observed_data, observed_mask, observed_tp, gt_mask in new_dataloader:
            optimizer.zero_grad()

            # Forward — signature identik HWD.forward()
            loss = model(observed_data, observed_mask, observed_tp, gt_mask)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        elapsed  = time.time() - epoch_time

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:>3}/{epochs} | "
                  f"loss {avg_loss:.6f} | {elapsed:.1f}s")

    # 4. Unfreeze semua untuk inference
    model = unfreeze_all(model)
    print(f"\nFine-tuning selesai. Semua layer di-unfreeze untuk inference.")

    return model


def evaluate_forgetting(model, old_dataloader, device=None):
    """
    Ukur catastrophic forgetting: evaluasi MAE di data lama
    setelah fine-tuning. Bandingkan dengan MAE sebelum fine-tuning.

    Returns:
        mae : float — MAE di data lama
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_err = 0.0
    total_pts = 0

    with torch.no_grad():
        for observed_data, observed_mask, observed_tp, gt_mask in old_dataloader:
            output = model.evaluate(
                observed_data, observed_mask, observed_tp, gt_mask,
                n_samples=10   # sedikit sample untuk efisiensi
            )
            imputed_samples, c_target, eval_points, _, _ = output

            # Median prediksi
            pred     = imputed_samples.median(dim=1).values  # [B, K, L]
            pred     = pred.permute(0, 2, 1)                 # [B, L, K]
            ev_mask  = eval_points.permute(0, 2, 1)

            total_err += torch.abs((pred - c_target) * ev_mask).sum().item()
            total_pts += ev_mask.sum().item()

    mae = total_err / (total_pts + 1e-8)
    return mae


def run_forgetting_check(base_ckpt_path, new_ckpt_path,
                         configs, old_dataloader, new_dataloader):
    """
    Helper lengkap untuk cek catastrophic forgetting.
    Print ringkasan perbandingan sebelum dan sesudah fine-tuning.

    Contoh dari notebook:
        import incremental
        incremental.run_forgetting_check(
            base_ckpt_path = 'checkpoints/hwd_2020_2025.pt',
            new_ckpt_path  = 'checkpoints/hwd_2020_2026_incr.pt',
            configs        = cfg,
            old_dataloader = old_test_loader,
            new_dataloader = new_test_loader,
        )
    """
    from models import main_model

    device = configs.device

    # Model lama
    model_old = main_model.HWD(configs).to(device)
    model_old.load_state_dict(
        torch.load(base_ckpt_path, map_location=device)
    )

    # Model baru (setelah incremental)
    model_new = main_model.HWD(configs).to(device)
    model_new.load_state_dict(
        torch.load(new_ckpt_path, map_location=device)
    )

    print("\n── Forgetting check ──")

    # MAE di data lama
    mae_old_on_old = evaluate_forgetting(model_old, old_dataloader)
    mae_new_on_old = evaluate_forgetting(model_new, old_dataloader)
    forgetting_pct = (mae_new_on_old - mae_old_on_old) / (mae_old_on_old + 1e-8) * 100

    # MAE di data baru
    mae_old_on_new = evaluate_forgetting(model_old, new_dataloader)
    mae_new_on_new = evaluate_forgetting(model_new, new_dataloader)
    improvement_pct = (mae_old_on_new - mae_new_on_new) / (mae_old_on_new + 1e-8) * 100

    print(f"  Data lama — MAE sebelum : {mae_old_on_old:.6f}")
    print(f"  Data lama — MAE sesudah : {mae_new_on_old:.6f}")
    print(f"  Forgetting              : {forgetting_pct:+.2f}%  (target < 5%)")
    print()
    print(f"  Data baru — MAE sebelum : {mae_old_on_new:.6f}")
    print(f"  Data baru — MAE sesudah : {mae_new_on_new:.6f}")
    print(f"  Improvement             : {improvement_pct:+.2f}%  (target > 0%)")
    print("──────────────────────")

    return {
        "mae_old_on_old"  : mae_old_on_old,
        "mae_new_on_old"  : mae_new_on_old,
        "forgetting_pct"  : forgetting_pct,
        "mae_old_on_new"  : mae_old_on_new,
        "mae_new_on_new"  : mae_new_on_new,
        "improvement_pct" : improvement_pct,
    }