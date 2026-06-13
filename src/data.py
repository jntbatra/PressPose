"""Data pipeline for cross-modal pairing.

Two sources:
  * real_pairs()      -- align insole + openpose CSVs by wall-clock second,
                         window them into fixed-length sequences, and emit
                         PAIRED / UNPAIRED samples.
  * SyntheticPairs    -- generates correlated/uncorrelated sensor sequences so
                         the full pipeline can be smoke-tested without the full
                         dataset (clearly synthetic; not for reporting metrics).
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

INSOLE_COLS = ["R_HEEL", "R_THUMB", "R_INNER_BALL", "R_OUTER_BALL",
               "L_HEEL", "L_THUMB", "L_INNER_BALL", "L_OUTER_BALL"]

SKEL_COLS = ["HIP_RIGHT_X", "HIP_RIGHT_Y", "HIP_LEFT_X", "HIP_LEFT_Y",
             "KNEE_RIGHT_X", "KNEE_RIGHT_Y", "KNEE_LEFT_X", "KNEE_LEFT_Y",
             "ANKLE_RIGHT_X", "ANKLE_RIGHT_Y", "ANKLE_LEFT_X", "ANKLE_LEFT_Y"]


def _resample(arr, length):
    """Linearly resample a (n, d) array to (length, d) along axis 0."""
    n = len(arr)
    if n == length:
        return arr
    src = np.linspace(0, 1, n)
    dst = np.linspace(0, 1, length)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def _hms(t):
    """Normalise a TIME field to 'H:M:S' (openpose has a trailing frame index)."""
    return ":".join(str(t).split(":")[:3])


def load_aligned(insole_csv, openpose_csv, seq_len=10):
    """Return aligned (insole_seqs, skel_seqs): lists of (seq_len, d) arrays,
    one per shared second."""
    ins = pd.read_csv(insole_csv)
    ops = pd.read_csv(openpose_csv)
    ins["_sec"] = ins["TIME"].map(_hms)
    ops["_sec"] = ops["TIME"].map(_hms)

    insole_seqs, skel_seqs = [], []
    for sec in sorted(set(ins["_sec"]) & set(ops["_sec"])):
        ib = ins.loc[ins["_sec"] == sec, INSOLE_COLS].to_numpy(dtype=np.float32)
        sb = ops.loc[ops["_sec"] == sec, SKEL_COLS].to_numpy(dtype=np.float32)
        if len(ib) == 0 or len(sb) == 0:
            continue
        insole_seqs.append(_resample(ib, seq_len).astype(np.float32))
        skel_seqs.append(_resample(sb, seq_len).astype(np.float32))
    return insole_seqs, skel_seqs


def _normalise(seqs):
    x = np.stack(seqs)
    mu, sd = x.mean((0, 1), keepdims=True), x.std((0, 1), keepdims=True) + 1e-6
    return [(s - mu[0]) / sd[0] for s in seqs]


class PairDataset(Dataset):
    """Builds balanced PAIRED/UNPAIRED samples from per-person aligned seqs.

    `people` is a dict: name -> (insole_seqs, skel_seqs) already aligned and
    index-matched (seq i of insole pairs with seq i of skeleton for that
    person). UNPAIRED samples cross insole of one person with skeleton of
    another.
    """

    def __init__(self, people, seed=0):
        self.samples = []  # (insole, skel, label)
        names = list(people)
        rng = np.random.default_rng(seed)

        for name in names:
            ins, skel = people[name]
            for i in range(min(len(ins), len(skel))):
                self.samples.append((ins[i], skel[i], 0))  # PAIRED

        if len(names) > 1:
            for name in names:
                ins, _ = people[name]
                others = [n for n in names if n != name]
                for i in range(len(ins)):
                    other = others[rng.integers(len(others))]
                    _, oskel = people[other]
                    j = rng.integers(len(oskel))
                    self.samples.append((ins[i], oskel[j], 1))  # UNPAIRED

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ins, skel, label = self.samples[idx]
        return (torch.from_numpy(ins).float(),
                torch.from_numpy(skel).float(),
                torch.tensor(label, dtype=torch.long))


def real_pairs(data_dir, seq_len=10):
    """Build a PairDataset from CSVs in `data_dir`.

    Convention: smart-insole-<X>.csv pairs with open-pose-<X>.csv. When file
    stems don't line up one-to-one, all insole files share the available
    openpose files (best-effort) -- intended for the full released dataset.
    """
    insole_files = sorted(glob.glob(os.path.join(data_dir, "smart-insole-*.csv")))
    op_files = sorted(glob.glob(os.path.join(data_dir, "open-pose-*.csv")))
    if not insole_files or not op_files:
        raise FileNotFoundError(f"No insole/openpose CSVs in {data_dir}")

    people = {}
    for k, icsv in enumerate(insole_files):
        ocsv = op_files[k % len(op_files)]
        name = os.path.splitext(os.path.basename(icsv))[0]
        ins, skel = load_aligned(icsv, ocsv, seq_len)
        if ins:
            people[name] = (_normalise(ins), _normalise(skel))
    return PairDataset(people)


class SyntheticPairs(Dataset):
    """Synthetic correlated (PAIRED) / independent (UNPAIRED) sequences.

    For development and CI only -- DO NOT report metrics from this.
    """

    def __init__(self, n=512, seq_len=10, insole_dim=8, skel_dim=12, seed=0):
        rng = np.random.default_rng(seed)
        # Fixed projections shared across the dataset so the PAIRED signal is
        # consistent and therefore learnable / generalisable.
        wi = rng.normal(size=(1, insole_dim)).astype(np.float32)
        ws = rng.normal(size=(1, skel_dim)).astype(np.float32)
        self.items = []
        for _ in range(n):
            label = int(rng.integers(2))
            base = rng.normal(size=(seq_len, 1)).astype(np.float32)
            ins = base * wi + 0.3 * rng.normal(size=(seq_len, insole_dim)).astype(np.float32)
            if label == 0:  # PAIRED -> skeleton shares the same latent base
                skel = base * ws + 0.3 * rng.normal(size=(seq_len, skel_dim)).astype(np.float32)
            else:           # UNPAIRED -> independent latent
                other = rng.normal(size=(seq_len, 1)).astype(np.float32)
                skel = other * ws + 0.3 * rng.normal(size=(seq_len, skel_dim)).astype(np.float32)
            self.items.append((ins, skel, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ins, skel, label = self.items[idx]
        return (torch.from_numpy(ins), torch.from_numpy(skel),
                torch.tensor(label, dtype=torch.long))
