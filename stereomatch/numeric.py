"""
Common numerical stuff.
"""

def is_power_of_two(num: int) -> bool:
    """
    https://stackoverflow.com/a/57027610
    """

    return (num != 0) and (num & (num-1) == 0)


def next_power_of_2(n: int) -> int:
    """
    Computes the next power of 2.
    """
    if n == 0:
        return 1
    if is_power_of_two(n):
        return n
    cnt = 0
    while n > 0:
        n = n >> 1
        cnt += 1

    return 1 << (cnt)
