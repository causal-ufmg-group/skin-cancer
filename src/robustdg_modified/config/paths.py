from pathlib import Path

CWD = Path().absolute()

CSV_DIR = CWD.parent / "data" / "csv_files"

LABELS_CSV = {
    "train": CSV_DIR / "ISIC2018_Task3_Training_GroundTruth.csv",
    "augmented_train": CSV_DIR / "Augmented_Training_GroundTruth.csv",
    "test": CSV_DIR / "Testing_GroundTruth.csv",
}

DOMAIN_TRAIN_CSV = {
    "train": CSV_DIR / "ISIC2018_Task3_Training_LesionGroupings.csv",
    "augmented_train": CSV_DIR / "Augmented_Training_Domain.csv",
}


IMG_DIR = {
    "train": CWD.parent / "data" / "ISIC2018_Task3_Training_Input/",
    "augmented_train": CWD.parent / "data" / "Augmented_Dataset/",
    "test": CWD.parent / "data" / "ISIC_2019_Training_Input/",
}

LOG_DIR = CWD.parent / "logs/"
CHECKPOINT_LOG_DIR = LOG_DIR / "checkpoints/"

CHECKPOINT_LOG_DIR.mkdir(parents=True, exist_ok=True)
