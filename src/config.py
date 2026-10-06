from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(r"C:\Users\Kinjal Chatterjee\Downloads\MILK10k")
IMAGE_DIR = DATA_DIR / "images"

METADATA_PATH = DATA_DIR / "metadata.csv"
GROUND_TRUTH_PATH = DATA_DIR / "supplements" / "training_gt.csv"

RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)

SPLIT_DIR = PROJECT_ROOT / "splits"
FIGURE_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
