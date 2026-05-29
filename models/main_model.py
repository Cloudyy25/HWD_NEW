import numpy as np
import torch
import torch.nn as nn
from models.diffusion import diff_HWD

# ─────────────────────────────────────────────────────────
# HWD  (menggantikan class FGTI di main_model.py FGTI asli)
#
# Perubahan dari FGTI asli:
#   1. Import diff_HWD bukan diff_FGTI
#   2. process_data: hapus observed_dataf (tidak pakai FFT lagi)
#   3. calc_loss: loss hanya di Ω (missing positions), bukan seluruh seq
#   4. set_input_to_diffmodel: hapus observed_dataf
#   5. impute / evaluate / forward: hapus observed_dataf
#
# Semua yang lain (noise schedule, time embedding, get_randmask,
# get_side_info, impute loop) identik dengan FGTI asli.
# ─────────────────────────────────────────────────────────
class HWD(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs        = configs
        self.device         = configs.device
        self.target_dim     = configs.enc_in

        self.emb_time_dim    = configs.timeemb
        self.emb_feature_dim = configs.featureemb
        self.emb_total_dim   = self.emb_time_dim + self.emb_feature_dim
        self.configs.side_dim = self.emb_total_dim

        self.embed_layer = nn.Embedding(
            num_embeddings=self.configs.enc_in,
            embedding_dim=self.emb_feature_dim
        )

        # ── HWD: pakai diff_HWD, bukan diff_FGTI ────────────────────
        self.diffmodel = diff_HWD(self.configs)

        # ── Noise schedule — identik FGTI ───────────────────────────
        self.num_steps = configs.diffusion_step_num
        if configs.schedule == "quad":
            self.beta = np.linspace(
                configs.beta_start ** 0.5,
                configs.beta_end ** 0.5,
                self.num_steps
            ) ** 2
        elif configs.schedule == "linear":
            self.beta = np.linspace(
                configs.beta_start, configs.beta_end, self.num_steps
            )
        self.alpha_hat   = 1 - self.beta
        self.alpha       = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float()
            .to(self.device)
            .unsqueeze(1).unsqueeze(1)
        )

    # ── Helpers — identik FGTI kecuali catatan di bawah ────────────

    def process_data(self, observed_data, observed_mask, observed_tp, gt_mask):
        """
        Hapus observed_dataf vs FGTI — tidak ada lagi FFT features.
        Signature: 4 argumen, bukan 5.
        """
        observed_data  = observed_data.to(self.device).float().permute(0, 2, 1)
        observed_mask  = observed_mask.to(self.device).float().permute(0, 2, 1)
        observed_tp    = observed_tp.to(self.device).float()
        gt_mask        = gt_mask.to(self.device).float().permute(0, 2, 1)
        return observed_data, observed_mask, observed_tp, gt_mask

    def time_embedding(self, pos, d_model=128):
        """Identik FGTI."""
        pe       = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0,
            torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        """Identik FGTI."""
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio  = np.random.rand()
            num_observed  = observed_mask[i].sum().item()
            num_masked    = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        """Identik FGTI."""
        B, K, L = cond_mask.shape
        time_embed    = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed    = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info     = torch.cat([time_embed, feature_embed], dim=-1)
        side_info     = side_info.permute(0, 3, 2, 1)   # [B, side_dim, K, L]
        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """
        Hapus observed_dataf vs FGTI — HWD tidak butuh FFT features di sini.
        """
        cond_obs     = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input  = torch.cat([cond_obs, noisy_target], dim=1)  # [B,2,K,L]
        return total_input

    # ── Loss — PERUBAHAN UTAMA ───────────────────────────────────────

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info):
        """
        Perubahan dari FGTI:
          - Hapus observed_dataf dari argumen dan dari diffmodel call
          - Loss dihitung HANYA di Ω (missing positions yang menjadi target),
            bukan di seluruh sequence.
            Ω = observed_mask - cond_mask
              = posisi yang ada datanya (observed) tapi sengaja di-mask
                untuk self-supervised training.
        """
        B, K, L = observed_data.shape

        diffusion_step = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha  = self.alpha_torch[diffusion_step]   # [B,1,1]

        noise      = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data \
                   + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        # cond_obs untuk conditioning di diff_HWD
        cond_obs  = cond_mask * observed_data               # [B, K, L]

        predicted = self.diffmodel(
            total_input, side_info, cond_obs, diffusion_step
        )                                                    # [B, K, L]

        # ── HWD: loss hanya di Ω ────────────────────────────────────
        # Ω = posisi observed yang sengaja di-mask (bukan seluruh missing)
        omega    = observed_mask - cond_mask                 # [B, K, L]
        residual = (noise - predicted) * omega
        num_eval = omega.sum()
        loss     = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    # ── Impute & Evaluate — hapus observed_dataf ────────────────────

    def impute(self, observed_data, cond_mask, side_info, n_samples=100):
        """Identik FGTI kecuali hapus observed_dataf dari semua call."""
        B, K, L        = observed_data.shape
        observed_data  = observed_data * cond_mask
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                total_input = self.set_input_to_diffmodel(
                    current_sample, observed_data, cond_mask
                )
                cond_obs  = cond_mask * observed_data

                predicted = self.diffmodel(
                    total_input, side_info,
                    cond_obs,
                    torch.tensor([t]).to(self.device)
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise  = torch.randn_like(current_sample)
                    sigma  = (
                        (1.0 - self.alpha[t - 1])
                        / (1.0 - self.alpha[t])
                        * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, observed_data, observed_mask, observed_tp, gt_mask):
        """Identik FGTI kecuali hapus observed_dataf."""
        (observed_data, observed_mask,
         observed_tp, gt_mask) = self.process_data(
            observed_data, observed_mask, observed_tp, gt_mask
        )
        observed_data = observed_data * observed_mask
        cond_mask     = self.get_randmask(observed_mask)
        side_info     = self.get_side_info(observed_tp, cond_mask)
        return self.calc_loss(observed_data, cond_mask, observed_mask, side_info)

    def evaluate(self, observed_data, observed_mask, observed_tp, gt_mask,
                 n_samples=100):
        """Identik FGTI kecuali hapus observed_dataf."""
        (observed_data, observed_mask,
         observed_tp, gt_mask) = self.process_data(
            observed_data, observed_mask, observed_tp, gt_mask
        )
        with torch.no_grad():
            cond_mask = observed_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            imputed_samples = self.impute(
                observed_data, cond_mask, side_info, n_samples
            )
            # return sama dengan FGTI: samples, data, eval_mask, obs_mask, tp
            return (
                imputed_samples,
                observed_data,
                gt_mask - observed_mask,
                observed_mask,
                observed_tp
            )