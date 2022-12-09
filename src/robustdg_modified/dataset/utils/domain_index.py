import pandas as pd

from .one_hot_encoding import convert_one_hot_df_to_names


def get_domain_index(domain_one_hot_df: pd.DataFrame) -> pd.DataFrame:

    """
    Given a pd.DataFrame with all one-hot encoded domain values,
    returns a separate indexes for each of them.

    ----
    Parameters:

        domain_one_hot: pd.DataFrame

            pd.DataFrame with one-hot encoded domain information.

    ----
    Returns:

        pd.DataFrame:

            DataFrame containing domain specific index, that it, an
            index in the range [0, len(d)) for each domain d.

    ----
    Example:

        Input: pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],  # df index
                "domain": [
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                ]
            }
        )

        Output: pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],
                "domain": [
                    [0, 0, 1],  # 0-th index for third domain
                    [0, 1, 0],  # 0-th index for second domain
                    [0, 0, 1],  # 1-st index for third domain
                    [1, 0, 0],  # 0-th index for first domain
                    [1, 0, 0],  # 1-st index for first domain
                ]
                "domain index": [0, 0, 1, 0, 1]
            }
        )
    """

    def create_domain_index(domain_filtered: pd.DataFrame) -> pd.DataFrame:

        size_domain = len(domain_filtered)

        index = domain_filtered.index
        data = list(range(size_domain))

        return pd.Series(data=data, index=index)

    id_to_domain = convert_one_hot_df_to_names(domain_one_hot_df, "domain").to_frame()

    return (
        id_to_domain.groupby("domain", group_keys=False)
        .apply(create_domain_index)
        .sort_index()
        .rename("domain index")
    )
