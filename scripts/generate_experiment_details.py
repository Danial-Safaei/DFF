#!/usr/bin/env python
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
PAIRS_JSONL = os.path.join(ROOT, "vkitti_baseline_pairs.jsonl")

SCENE_SPLITS = {
    "calibration": ["Scene01", "Scene02", "Scene06"],
    "heldout": ["Scene18", "Scene20"],
}

SEQ2SCENE = {
    "0001": "Scene01",
    "0002": "Scene02",
    "0006": "Scene06",
    "0018": "Scene18",
    "0020": "Scene20",
}

TAU_OVF = 0.7
TAU_DFF_DISTANCE_PER_SUT = {
    "steering": {"eps_90": 0.064245, "eps_95": 0.08738442},
    "da": {"eps_90": 0.570196, "eps_95": 0.6990833},
    "ll": {"eps_90": 0.582134, "eps_95": 0.7362408},
}

EPSILON_STEERING = 0.005
EPSILON_SEGMENTATION = 0.010

KITTI_SEQ_RE = re.compile(r"/KITTI/(\d{4})/")
KITTI_FRAME_RE = re.compile(r"/(\d{6})\.png$")
VKITTI_SCENE_RE = re.compile(r"/Scene(\d{2})/")
VKITTI_FRAME_RE = re.compile(r"/classgt_(\d{5})\.png$")


def find_latest(pattern: str, exclude: Optional[Iterable[str]] = None) -> Optional[str]:
    matches = glob.glob(pattern)
    if exclude:
        matches = [m for m in matches if all(x not in os.path.basename(m) for x in exclude)]
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def split_for_scene(scene: str) -> str:
    for split, scenes in SCENE_SPLITS.items():
        if scene in scenes:
            return split
    return "unknown"


@dataclass
class PairRow:
    pair_id: str
    kitti_path: str
    vkitti_path: str
    scene: str
    frame: int
    split: str


def parse_pair(item: Dict) -> Optional[PairRow]:
    kitti_path = item.get("image")
    vkitti_path = item.get("conditioning_image")
    if not kitti_path or not vkitti_path:
        return None

    seq_match = KITTI_SEQ_RE.search(kitti_path)
    frame_match = KITTI_FRAME_RE.search(kitti_path)
    scene_match = VKITTI_SCENE_RE.search(vkitti_path)
    seg_frame_match = VKITTI_FRAME_RE.search(vkitti_path)

    if not (seq_match and frame_match and scene_match and seg_frame_match):
        return None

    seq = seq_match.group(1)
    frame = int(frame_match.group(1))
    scene = f"Scene{scene_match.group(1)}"
    seg_frame = int(seg_frame_match.group(1))

    if frame % 100000 != seg_frame:
        pass

    split = split_for_scene(scene)
    pair_id = f"{seq}_{frame:06d}"
    return PairRow(pair_id, kitti_path, vkitti_path, scene, frame, split)


def load_pairs() -> List[PairRow]:
    rows: List[PairRow] = []
    with open(PAIRS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            parsed = parse_pair(item)
            if parsed:
                rows.append(parsed)
    return rows


def write_pairs_csv(rows: List[PairRow], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "kitti_path", "vkitti_path", "scene", "frame", "split"])
        for r in rows:
            writer.writerow([r.pair_id, r.kitti_path, r.vkitti_path, r.scene, r.frame, r.split])


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return center, max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 12345) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(values)), float(lo), float(hi)


def compute_spearman(df: pd.DataFrame, dff_col: str) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for sut in ["steering", "da", "ll"]:
        sut_df = df[df["SUT"] == sut].dropna(subset=["InputFid", "OVF", dff_col])
        if len(sut_df) < 5:
            continue
        rho_iv_ov, _ = stats.spearmanr(sut_df["InputFid"], sut_df["OVF"])
        rho_ov_dff, _ = stats.spearmanr(sut_df["OVF"], sut_df[dff_col])
        n = len(sut_df)
        z = 0.5 * np.log((1 + rho_ov_dff) / (1 - rho_ov_dff + 1e-10))
        se = 1 / np.sqrt(max(1, n - 3))
        z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
        ci_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
        ci_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
        rows.append(
            {
                "metric": "spearman",
                "SUT": sut,
                "rho_iv_ov": float(rho_iv_ov),
                "rho_ov_dff": float(rho_ov_dff),
                "rho_ov_dff_ci_lo": float(ci_lo),
                "rho_ov_dff_ci_hi": float(ci_hi),
                "n": int(n),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pairs = load_pairs()
    pairs_csv = os.path.join(ROOT, "pairs.csv")
    write_pairs_csv(pairs, pairs_csv)

    rq1_csv = find_latest(os.path.join(RESULTS_DIR, "rq1_multi_*.csv"), exclude=["agg", "probability"])
    if rq1_csv is None:
        raise FileNotFoundError("No rq1_multi_*.csv found in results/")

    df = pd.read_csv(rq1_csv)
    if "seq" in df.columns:
        df["seq"] = df["seq"].astype(str).str.zfill(4)
    if "frame" in df.columns:
        df["frame"] = df["frame"].astype(int, errors="ignore")
    dff_col = "DFF_distance" if "DFF_distance" in df.columns else "RFF_distance"

    pairs_df = pd.DataFrame([r.__dict__ for r in pairs])
    pairs_df["seq"] = pairs_df["pair_id"].str.split("_").str[0].astype(str).str.zfill(4)
    pairs_df["frame"] = pairs_df["frame"].astype(int)

    merged = df.merge(pairs_df, how="left", left_on=["seq", "frame"], right_on=["seq", "frame"])
    merged.rename(columns={"vkitti_path": "vkitti_path"}, inplace=True)
    metrics_out = os.path.join(ROOT, "metrics_results.csv")
    merged.to_csv(metrics_out, index=False)

    stats_rows = []

    spearman_df = compute_spearman(df, dff_col)
    if not spearman_df.empty:
        stats_rows.append(spearman_df)

    # Pass rates for eps90 and eps95
    pass_rows = []
    for (sut, gen), grp in df.groupby(["SUT", "generator"]):
        valid = grp.dropna(subset=["OVF", dff_col])
        n = len(valid)
        if n == 0:
            continue
        for percentile in [90, 95]:
            eps = TAU_DFF_DISTANCE_PER_SUT.get(sut, {}).get(f"eps_{percentile}", 0.3)
            k = int(((valid["OVF"] >= TAU_OVF) & (valid[dff_col] <= eps)).sum())
            p_hat, ci_lo, ci_hi = wilson_ci(k, n)
            pass_rows.append(
                {
                    "metric": "pass_rate",
                    "SUT": sut,
                    "generator": gen,
                    "percentile": percentile,
                    "eps": eps,
                    "n": int(n),
                    "k": int(k),
                    "p_hat": float(p_hat),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                }
            )
    if pass_rows:
        stats_rows.append(pd.DataFrame(pass_rows))

    # Non-inferiority deltas: DFF vs Baseline
    ni_rows = []
    for sut in ["steering", "da", "ll"]:
        base = df[(df["SUT"] == sut) & (df["generator"] == "Baseline")]
        dff_candidates = df[(df["SUT"] == sut) & (df["generator"].str.contains("DFF"))]
        if dff_candidates.empty or base.empty:
            continue
        dff = dff_candidates
        # pair on seq+frame
        base_key = base.set_index(["seq", "frame"])
        dff_key = dff.set_index(["seq", "frame"])
        common = base_key.index.intersection(dff_key.index)
        if len(common) == 0:
            continue
        base_common = base_key.loc[common]
        dff_common = dff_key.loc[common]

        for metric in ["OVF", "InputFid"]:
            diffs = (dff_common[metric].values - base_common[metric].values).astype(float)
            mean, ci_lo, ci_hi = bootstrap_ci(diffs, n_boot=1000, seed=12345)
            epsilon = EPSILON_STEERING if sut == "steering" else EPSILON_SEGMENTATION
            ni_pass = ci_lo > -epsilon
            ni_rows.append(
                {
                    "metric": "ni_delta",
                    "SUT": sut,
                    "metric_name": metric,
                    "n": int(len(diffs)),
                    "delta_mean": float(mean),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                    "epsilon": float(epsilon),
                    "ni_pass": bool(ni_pass),
                }
            )
    if ni_rows:
        stats_rows.append(pd.DataFrame(ni_rows))

    stats_df = pd.concat(stats_rows, ignore_index=True) if stats_rows else pd.DataFrame()
    stats_out = os.path.join(ROOT, "stats_summary.csv")
    stats_df.to_csv(stats_out, index=False)

    print("pairs_csv=", pairs_csv)
    print("metrics_csv=", metrics_out)
    print("stats_csv=", stats_out)
    print("source_rq1=", rq1_csv)


if __name__ == "__main__":
    main()
