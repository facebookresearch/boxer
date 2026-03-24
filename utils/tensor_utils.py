# pyre-unsafe
from bisect import bisect_left

import numpy as np
import torch


def find_nearest(array, value, return_index=False, warning_if_beyond=False):
    """Find the index of the nearest value in a numpy array."""
    array = np.asarray(array)
    val = np.abs(array - value)
    idx = (val).argmin()
    if return_index:
        return idx
    else:
        return array[idx]


def find_nearest2(array, value):
    """Find the index of the nearest value in an array."""
    idx = bisect_left(array, value)
    if idx == len(array):
        return idx - 1
    if idx == 0:
        return 0
    before = array[idx - 1]
    after = array[idx]
    if after - value < value - before:
        return idx
    return idx - 1


def pad_string(string, max_len=200, silent=False):
    """Pad a string with "spaces" at the end, up to 200 by default."""
    string2 = string[:max_len]
    if len(string) > max_len and not silent:
        print("Warning: string will be truncated to %s" % string2)
    if len(string) > 0 and string2[-1] == " " and not silent:
        print("Warning: string ends with a space, this may be lost when unpadding")
    return "{message: <{width}}".format(message=string2, width=max_len)


def unpad_string(string, max_len=200):
    """Remove extra space padding at the end of the string."""
    return string.rstrip()


def string2tensor(string):
    """convert a python string into torch tensor of chars"""
    return torch.tensor([ord(s) for s in string]).byte()


def tensor2string(tensor, unpad=False):
    """convert a torch tensor of chars to python string"""

    def safe_chr(val):
        """Convert integer to character, handling invalid values"""
        try:
            # Clamp to valid Unicode range
            val = int(val)
            if val < 0 or val > 0x10FFFF:
                return ""  # Skip invalid characters
            return chr(val)
        except (ValueError, OverflowError):
            return ""  # Skip on error

    if tensor.ndim == 1:
        out = "".join([safe_chr(s) for s in tensor])
        if unpad:
            out = unpad_string(out)
        return out
    elif tensor.ndim == 2:
        out = []
        for ex in tensor:
            ex2 = "".join([safe_chr(s) for s in ex])
            if unpad:
                ex2 = unpad_string(ex2)
            out.append(ex2)
        return out
    else:
        raise ValueError("Higher dims >2 not supported")


def pad_points(points_in, max_num_point=25000):
    """Pad point matrix with nan at the end, return fixed size matrix.
    the last row will be:
    nan nan nan ... numValidRow
    """
    assert max_num_point >= 3

    if points_in.ndim == 1:
        points_in = points_in.reshape(-1, 3)

    points_padded = torch.zeros(
        (max_num_point, points_in.shape[1]), device=points_in.device
    )
    numValidRow = min(points_in.shape[0], max_num_point - 1)

    points_padded[0:numValidRow, :] = points_in[0:numValidRow, :]
    points_padded[numValidRow:, :] = float("nan")  # all nan from numValidRow
    points_padded[-1, -1] = numValidRow
    return points_padded
