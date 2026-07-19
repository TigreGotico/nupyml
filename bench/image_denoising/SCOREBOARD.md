# Scoreboard — image_denoising

**Goal:** Denoise noisy images; scored by mean PSNR (dB) against the clean originals.

**Metric:** `psnr_db` (higher is better) · **QA floor:** `20.0`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_non_local_means` | 24.8594 | 1.00 |
| 2 | `baseline_total_variation` | 23.5075 | 0.71 |
| 3 | `baseline_bilateral` | 22.4900 | 0.63 |

_Regenerate with `python bench/harness.py image_denoising`._
