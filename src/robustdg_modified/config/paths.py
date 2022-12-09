from pathlib import Path

CWD = Path().absolute()

LABELS_CSV = {
    "train": (
        CWD.parent
        / "data"
        / "ISIC2018_Task3_Training_GroundTruth"
        / "ISIC2018_Task3_Training_GroundTruth.csv"
    ),
    "augmented_train": (
        CWD.parent / "data" / "Augmented_Dataset" / "Augmented_Training_GroundTruth.csv"
    ),
    "test": (
        CWD.parent
        / "data"
        / "ISIC2018_Task3_Validation_GroundTruth"
        / "ISIC2018_Task3_Validation_GroundTruth.csv"
    ),
}

DOMAIN_TRAIN_CSV = {
    "train": (
        CWD.parent
        / "data"
        / "ISIC2018_Task3_Training_GroundTruth"
        / "ISIC2018_Task3_Training_LesionGroupings.csv"
    ),
    "augmented_train": (
        CWD.parent
        / "data"
        / "Augmented_Dataset"
        / "Augmented_Training_Domain.csv"
    )
}


IMG_DIR = {
    "train": CWD.parent / "data" / "ISIC2018_Task3_Training_Input/",
    "augmented_train": CWD.parent / "data" / "Augmented_Dataset/",
    "test": CWD.parent / "data" / "ISIC2018_Task3_Validation_Input/",
}

LOG_DIR = CWD.parent / "logs/"
CHECKPOINT_LOG_DIR = LOG_DIR / "checkpoints/"

CHECKPOINT_LOG_DIR.mkdir(parents=True, exist_ok=True)
