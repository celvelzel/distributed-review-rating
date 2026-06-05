from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
ARTIFACTS_DIR = ROOT.parent / "artifacts"
OUTPUT_DIR = ROOT.parent / "output"

STAGES = list(range(7))
RANDOM_SEED = 42
