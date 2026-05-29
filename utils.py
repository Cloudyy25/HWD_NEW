"""
utils.py — HWD
File baru — tidak ada di FGTI asli.

Isi:
  - calc_quantile_CRPS     : copy langsung dari CSDI utils.py
  - calc_quantile_CRPS_sum : copy langsung dari CSDI utils.py
  - quantile_loss          : copy langsung dari CSDI utils.py
  - calc_denominator       : copy langsung dari CSDI utils.py

Semua fungsi identik CSDI. Hanya dipindahkan ke sini agar
bisa diimport dari train.py tanpa dependency ke CSDI lainnya.
"""

import numpy as np
import torch


# ── Identik CSDI utils.py ──────────────────────────────────────────

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs(
            (forecast - target) * eval_points
            * ((target <= forecast) * 1.0 - q)
        )
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points,
                       mean_scaler, scaler):
    """
    CRPS probabilistik via quantile loss.

    Args:
        target       : [B, L, K]            — ground truth
        forecast     : [B, n_samples, L, K] — sampel imputasi
        eval_points  : [B, L, K]            — mask Ω (1=evaluate, 0=skip)
        mean_scaler  : float — mean scaler untuk denormalisasi
        scaler       : float — std scaler untuk denormalisasi

    Returns:
        CRPS : float (lebih kecil lebih baik)
    """
    target   = target   * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom     = calc_denominator(target, eval_points)
    CRPS      = 0

    for q in quantiles:
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(
                torch.quantile(forecast[j: j + 1], q, dim=1)
            )
        q_pred = torch.cat(q_pred, dim=0)
        q_loss = quantile_loss(target, q_pred, q, eval_points)
        CRPS  += q_loss / denom

    return CRPS.item() / len(quantiles)


def calc_quantile_CRPS_sum(target, forecast, eval_points,
                           mean_scaler, scaler):
    """
    CRPS-sum: CRPS pada jumlah semua fitur (untuk evaluasi joint).
    Opsional — tidak dipakai di evaluasi utama HWD tapi disertakan
    untuk kelengkapan komparasi dengan baseline CSDI.
    """
    eval_points = eval_points.mean(-1)
    target      = (target   * scaler + mean_scaler).sum(-1)
    forecast    =  forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom     = calc_denominator(target, eval_points)
    CRPS      = 0

    for q in quantiles:
        q_pred = torch.quantile(forecast.sum(-1), q, dim=1)
        q_loss = quantile_loss(target, q_pred, q, eval_points)
        CRPS  += q_loss / denom

    return CRPS.item() / len(quantiles)