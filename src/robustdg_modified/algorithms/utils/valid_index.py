import torch


def get_desired_entries_for_column(
    desired_pos: torch.Tensor, column: int, values_to_set: torch.Tensor = None
) -> torch.Tensor:

    """
    Returns boolean flattened desired_pos for a given column.

    ----
    Parameters:

        desired_pos: torch.Tensor

            Boolean tensor describing which entries are desired.
            Desired entries are True whereas invalid ones are False.

        column: int

            Column from which values should be returned.

            All values for other columns will be False.

        values_to_set: Tensor | None, default = None

            Values to be set to desired column.
    ----
    Returns:
        torch.Tensor:

            Flat tensor containing only desired_pos.

            All desired entries which don't belong to column will be False.
    ----
    Examples:

        Args:
            valid_index = [
                [ True,  True,  True],
                [ False, False,  True],
                [ True,  True, False]
            ]
            column = 0
            values_to_set = None

        Result:
            [
                True,  False, False,
                False,        False,
                True,  False
            ]  # one-dimensional tensor

            Note that undesired values from second and
            third row were removed.

        Args:
            valid_index = [
                [ True,  True,  True],
                [ False, False,  True],
                [ True,  True, False]
            ]
            column = 0
            values_to_set = [True, True, True]

        Result:
            [
                True, False, False,
                True,        False,
                True, False
            ]  # one-dimensional tensor

            Note that value at [1,0] is True, because the
            column was set to be so.

    """

    index = torch.zeros_like(desired_pos).bool()

    if values_to_set is None:
        values_to_set = desired_pos[:, column]

    index[:, column] = values_to_set

    return index[desired_pos].flatten()


def get_desired_entries_in_both_columns(
    desired_pos: torch.Tensor, first_col: int, second_col: int
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Returns boolean flattened desired_pos for each column.

    Desired positions in this case are the ones such that both
    input columns exist for the same row.

    ----
    Parameters:

        valid_index: torch.Tensor

            Tensor containing all indexes that are valid.

            valid_index[0] is an index that is valid for another tensor.

        <first,second>_col: int

            Desired columns numbers.

    ----
    Returns:
        tuple[torch.Tensor, torch.Tensor]:

            Respectively, mask for first and second columns.

    ----
    Example:

        Args:
            valid_index = [
                [ True,  True,  True],
                [ True, False,  True],
                [False,  True, False]
            ]
            first_col = 0
            second_col = 1

        Result:

            mask_first_col -> [
                True,  False, False,
                False,        False,
                       False
            ]  # one-dimensional tensor

            mask_second_col -> [
                False, True, False,
                False,       False,
                       False
            ]  # one-dimensional tensor

            Note that only the first row shows up for both first and second columns.
    """

    both_true = (desired_pos[:, first_col]) & (desired_pos[:, second_col])

    first_index = get_desired_entries_for_column(desired_pos, first_col, both_true)
    second_index = get_desired_entries_for_column(desired_pos, second_col, both_true)

    return first_index, second_index
