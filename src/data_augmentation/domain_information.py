import numpy as np
import pandas as pd


def get_information_per_domain_label(
    img_information: pd.DataFrame,
    column_names: tuple[str, str, str],
) -> pd.DataFrame:

    """
    Returns image names and size for each pair of domain and label.

    ----
    Parameters:

        img_information: pd.DataFrame[
            columns = [
                "{image name}": image names,\n
                "{domain name}": domain names,\n
                "{labels}": label names,\n
            ]
        ]

        column_names: tuple[str, str, str]
            Respectively, name for columns:
                image names, domain names, labels

    ----
    Returns:

        pd.DataFrame[
            index = [
                "{domain name}": domain names,\n
                "{labels}": label names,
            ]
            columns = [
                "names": numpy.array with all images names,
                "size": number of images,
            ]
        ]
    """
    img_names, domain_names, labels = column_names

    img_names_per_domain_label = img_information.groupby([domain_names, labels]).apply(
        lambda df: np.sort(df[img_names].to_numpy())
    )

    return pd.concat(
        [img_names_per_domain_label, img_names_per_domain_label.apply(len)],
        axis=1,
        keys=[img_names, "size"],
    )
