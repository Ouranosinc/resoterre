"""Type checker utilities."""


def type_check_slice(x: slice | None) -> slice:
    """
    Check if the input is a slice or None.

    Parameters
    ----------
    x : slice | None
        The input to check.

    Returns
    -------
    slice
        The input if it is a slice or None.

    Raises
    ------
    TypeError
        If the input is not a slice.
    """
    if x is None:
        raise TypeError("Expected a slice, got None")
    return x
