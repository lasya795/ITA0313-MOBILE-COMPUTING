"""Shared helpers: fixed seed, output folders, save helpers."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42                      # global fixed seed (reported in README)
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})


def rng(offset=0):
    """Reproducible generator: SEED + offset."""
    return np.random.default_rng(SEED + offset)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=150)
    plt.close(fig)
    print("  figure ->", name)


def save_csv(df, name):
    df.to_csv(os.path.join(DATA, name), index=False)
    print("  csv    ->", name)


def save_json(obj, name):
    with open(os.path.join(DATA, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)
