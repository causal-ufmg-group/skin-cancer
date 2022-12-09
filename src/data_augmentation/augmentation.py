import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional

import PIL.Image as Image


def copy_all_imgs(
    img_names: Iterable[str],
    from_dir: Path,
    to_dir: Path,
    img_extension: Optional[str] = None,
) -> None:

    """
    Move all images from "from_dir" to "to_dir".

    If "img_extension" is None, it will be assumed that
    each img name already has correct extension.
    """

    if img_extension is not None:
        img_names = [f"{img_name}.{img_extension}" for img_name in img_names]

    for img_name in img_names:

        from_path = from_dir / img_name
        to_path = to_dir / img_name
        shutil.copy(from_path, to_path)


def augment_all_imgs(
    img_names: Iterable[str],
    from_dir: Path,
    augmentation: Callable[[Image.Image], Image.Image],
    to_dir: Optional[Path] = None,
    img_extension: Optional[str] = None,
    suffix: Optional[str] = "_aug",
) -> None:

    """
    Augments images from "from_dir" and saves them to "to_dir".
    New img_name will be f"{img_name}{suffix}{extension}"

    If "to_dir" is not provided, it will be saved to the same
    folder.

    If "img_extension" is None, it will be assumed that
    each img name already has correct extension.

    """
    to_dir = to_dir if to_dir is not None else from_dir

    if img_extension is not None:
        img_names = [f"{img_name}.{img_extension}" for img_name in img_names]

    for img_name in img_names:

        pil_img = Image.open(from_dir / img_name)
        aug_img = augmentation(pil_img)

        name, extension = img_name.split(".")
        aug_img_filename = f"{name}{suffix}.{extension}"
        aug_img.save(to_dir / aug_img_filename)
