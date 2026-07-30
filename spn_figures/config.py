from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "source"
DERIVED = DATA / "derived"
FIGURES = ROOT / "figures"

SPN_NAMES = ["dSPN_left", "dSPN_right", "iSPN_left", "iSPN_right"]
SPN_BITS = {name: 2 ** (3 - i) for i, name in enumerate(SPN_NAMES)}
SPN_COLORS = {
    "dSPN_left": "#8B0000",
    "iSPN_left": "#F08080",
    "dSPN_right": "#000080",
    "iSPN_right": "#ADD8E6",
}
SPN_LINESTYLES = {
    "dSPN_left": "-",
    "dSPN_right": "-",
    "iSPN_left": "--",
    "iSPN_right": "--",
}

LAST_K = 6
BIN_SIZE_S = 0.010
BASELINE_WINDOW_S = (-0.200, 0.000)
BASELINE_ADD_HZ = 1.0
SMOOTH_SIGMA_S = 0.025
STAGE_B_WINDOW_S = (-0.100, 0.050)
RANDOM_SEED = 20
N_BOOTSTRAP = 5000
