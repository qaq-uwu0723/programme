"""Evaluation metrics: KS, JSD, Lag-1 Diff."""
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def compute_ks(
    real: np.ndarray,
    gen: np.ndarray,
    per_feature: bool = False,
) -> Dict:
    """Compute KS statistic per feature.

    Args:
        real: (N, L, d) or (N, d) real data
        gen:  (M, L, d) or (M, d) generated data

    Returns:
        dict with mean_ks, per_feature_ks list
    """
    if real.ndim == 3:
        real = real.reshape(-1, real.shape[-1])
    if gen.ndim == 3:
        gen = gen.reshape(-1, gen.shape[-1])

    d = real.shape[1]
    ks_vals = []
    for i in range(d):
        ks_i, _ = stats.ks_2samp(real[:, i], gen[:, i])
        ks_vals.append(ks_i)

    result = {
        "mean_ks": float(np.mean(ks_vals)),
        "max_ks": float(np.max(ks_vals)),
        "min_ks": float(np.min(ks_vals)),
    }
    if per_feature:
        result["per_feature"] = [float(v) for v in ks_vals]
    return result


def compute_jsd(
    real: np.ndarray,
    gen: np.ndarray,
    vocab_sizes: List[int],
    per_feature: bool = False,
) -> Dict:
    """Compute Jensen-Shannon Divergence per discrete variable.

    Args:
        real: (N, L, d_d) or (N, d_d) real discrete data
        gen:  (M, L, d_d) or (M, d_d) generated discrete data
        vocab_sizes: list of vocabulary sizes per variable

    Returns:
        dict with mean_jsd, per_feature_jsd
    """
    if real.ndim == 3:
        real = real.reshape(-1, real.shape[-1])
    if gen.ndim == 3:
        gen = gen.reshape(-1, gen.shape[-1])

    d_d = real.shape[1]
    eps = 1e-10
    jsd_vals = []
    for i in range(d_d):
        vs = vocab_sizes[i]
        # Use max of vocab size and observed values to avoid truncation
        max_val = max(int(real[:, i].max()), int(gen[:, i].max()), vs - 1)
        vs_eff = max_val + 1
        real_freq = np.bincount(real[:, i].astype(int), minlength=vs_eff).astype(float)
        gen_freq = np.bincount(gen[:, i].astype(int), minlength=vs_eff).astype(float)
        real_prob = real_freq / (real_freq.sum() + eps) + eps
        gen_prob = gen_freq / (gen_freq.sum() + eps) + eps
        real_prob /= real_prob.sum()
        gen_prob /= gen_prob.sum()
        # JSD = 0.5 * (KL(P||M) + KL(Q||M)) where M = (P+Q)/2
        m_prob = 0.5 * (real_prob + gen_prob)
        kl_pm = np.sum(real_prob * np.log2(real_prob / m_prob))
        kl_qm = np.sum(gen_prob * np.log2(gen_prob / m_prob))
        jsd = 0.5 * (kl_pm + kl_qm)
        jsd_vals.append(float(jsd if np.isfinite(jsd) else 0.0))

    result = {
        "mean_jsd": float(np.mean(jsd_vals)),
        "max_jsd": float(np.max(jsd_vals)),
    }
    if per_feature:
        result["per_feature"] = [float(v) for v in jsd_vals]
    return result


def compute_lag1_diff(
    real: np.ndarray,
    gen: np.ndarray,
    per_feature: bool = False,
) -> Dict:
    """Compute Lag-1 autocorrelation difference.

    Args:
        real: (N, L, d) real data
        gen:  (M, L, d) generated data

    Returns:
        dict with mean_lag1_diff, per_feature_lag1_diff
    """
    d = real.shape[2] if real.ndim == 3 else real.shape[1]

    def lag1_corr(data: np.ndarray) -> np.ndarray:
        """Compute lag-1 autocorrelation per feature."""
        if data.ndim == 3:
            # (N, L, d) — average over samples
            corrs = []
            for n in range(data.shape[0]):
                corr_n = []
                for i in range(d):
                    seq = data[n, :, i]
                    if np.std(seq) < 1e-9:
                        corr_n.append(0.0)
                    else:
                        corr_n.append(np.corrcoef(seq[:-1], seq[1:])[0, 1])
                corrs.append(corr_n)
            return np.nanmean(corrs, axis=0)
        else:
            # (N, d)
            return np.array([
                np.corrcoef(data[:-1, i], data[1:, i])[0, 1]
                if np.std(data[:, i]) > 1e-9 else 0.0
                for i in range(d)
            ])

    real_lag1 = lag1_corr(real)
    gen_lag1 = lag1_corr(gen)
    diffs = np.abs(real_lag1 - gen_lag1)

    result = {
        "mean_lag1_diff": float(np.mean(diffs)),
        "max_lag1_diff": float(np.max(diffs)),
    }
    if per_feature:
        result["per_feature"] = [float(v) for v in diffs]
    return result


def evaluate_all(
    real_x: np.ndarray,
    gen_x: np.ndarray,
    real_y: np.ndarray,
    gen_y: np.ndarray,
    vocab_sizes: List[int],
) -> Dict:
    """Run full evaluation suite.

    Args:
        real_x, gen_x: continuous data (N, L, d_c)
        real_y, gen_y: discrete data (N, L, d_d)
        vocab_sizes: vocab sizes for discrete variables

    Returns:
        dict with ks, jsd, lag1 metrics
    """
    ks_result = compute_ks(real_x, gen_x, per_feature=True)
    lag1_result = compute_lag1_diff(real_x, gen_x, per_feature=True) if real_x.ndim == 3 else {}
    jsd_result = compute_jsd(real_y, gen_y, vocab_sizes, per_feature=True)
    return {
        "ks": ks_result,
        "jsd": jsd_result,
        "lag1_diff": lag1_result,
    }
