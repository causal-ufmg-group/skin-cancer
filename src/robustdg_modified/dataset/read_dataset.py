import pandas as pd


def get_image_names(
    labels_csv: pd.DataFrame, img_names_col: str = "image"
) -> pd.Series:

    """
    Returns mapping of integer to image names, sorted by image names.

    -----
    Parameters:

        labels_csv: pd.DataFrame

            DataFrame should have a column for image names named "image".

        img_names_col: str

            Name of column with image names.

    -----
    Returns:

        pd.Series
            Sorted int to image name mapping.
    """
    sorted_df = labels_csv.sort_values(img_names_col).reset_index(drop=True)

    return sorted_df[img_names_col]


def get_one_hot_labels(
    labels_csv: pd.DataFrame, img_names_col: str = "image"
) -> pd.DataFrame:

    """
    Returns one-hot encoded labels for all images, sorted by image names.

    -----
    Parameters:

        labels_csv: pd.DataFrame

            DataFrame should have:
                1) a column for image names named "image"

                2) all other columns should be images one-hot encoded labels

        img_names_col: str

            Name of column with image names.

    -----
    Returns:

        pd.DataFrame
            One-hot encoded labels for all images, sorted by image names.
    """

    sorted_df = labels_csv.sort_values(img_names_col).reset_index(drop=True)

    return sorted_df.drop(img_names_col, axis=1)


def get_one_hot_domain(
    domain_csv: pd.DataFrame,
    img_names_col: str = "image",
    domain_col: str = "diagnosis_confirm_type",
) -> pd.DataFrame:

    """
    Returns one-hot encoded domain for all images, sorted by image names.

    -----
    Parameters:

        domain_csv: pd.DataFrame

            DataFrame should have:
                1) a column for image names named "image"

                2) a column with domain names for each image

        img_names_col: str

            Name of column with image names.

        domain_col: str

            Name of column with domain names.

    -----
    Returns:

        pd.DataFrame
            One-hot encoded labels for all images, sorted by image names.
    """

    sorted_df = domain_csv.sort_values(img_names_col).reset_index(drop=True)
    img_domain_names = sorted_df[domain_col]

    return pd.get_dummies(img_domain_names)
