import torch

def dice_coeff(result: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : torch.Tensor
        Input data containing objects. Should be a torch tensor with dtype torch.bool,
        where 0 represents background and 1 represents the object.
    reference : torch.Tensor
        Input data containing objects. Should be a torch tensor with dtype torch.bool,
        where 0 represents background and 1 represents the object.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    if isinstance(result, torch.FloatTensor) or result.dtype == torch.float:
        intersection = torch.sum(result.bool() & reference.bool()).item()
    elif isinstance(result, torch.LongTensor) or result.dtype == torch.int:
        intersection = torch.sum(result & reference).item()

    size_i1 = torch.sum(result).item()
    size_i2 = torch.sum(reference).item()

    try:
        dc_value = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc_value = 0.0

    return torch.tensor(dc_value)