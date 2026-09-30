"""Number Theoretic Transform over the fixed prime p = 998244353.

This prime admits roots of unity of order 2**23, which means the largest
power-of-two length the convolution can use is 2**23. Real inputs up to a
few thousand bits long are well within that range.

The library supports *one* transform modulus. Multiple-modulus NTT for
arbitrary-precision convolution is a real technique but a much larger API;
we pick the simple, fast, single-prime interpretation and state it plainly.
"""

_P = 998244353
_G = 3  # primitive root of p


def _pow_int(base: int, exp: int, mod: int) -> int:
    """Integer exponentiation by squaring.

    Builtins include pow(a, b, m) but for teaching-clarity this module
    keeps the operation explicit so the math is self-contained.
    """
    if exp < 0:
        raise ValueError("negative exponent not supported")
    result = 1
    base %= mod
    while exp:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result


def _bit_reverse(x: int, bits: int) -> int:
    """Reverse the low `bits` bits of x.

    Standard in-place iterative NTT needs this index permutation so that the
    butterfly passes read the correct pairs. Burying it inside the transform
    makes the loop harder to read; extracting it makes the intent obvious.
    """
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _primitive_root_of_unity(order: int) -> int:
    """Return a primitive `order`-th root of unity modulo p.

    p - 1 = 2**23 * 119, so for any order dividing 2**23 the generator is
    g ** ((p-1)/order) mod p. Using g = 3 (a known primitive root of p)
    keeps this a single closed-form computation with no search.
    """
    if order < 1 or (order & (order - 1)) != 0:
        raise ValueError("order must be a positive power of two")
    if (_P - 1) % order != 0:
        raise ValueError("order does not divide p-1; no root exists")
    return _pow_int(_G, (_P - 1) // order, _P)


def _transform(a: list, invert: bool) -> list:
    """In-place iterative NTT (or inverse NTT if `invert`).

    Requires len(a) to be a power of two and a divisor of 2**23.
    Returns the same list mutated; the return value is for convenience only.
    """
    n = len(a)
    if n == 0:
        return a
    if n & (n - 1) != 0:
        raise ValueError("length must be a power of two")
    bits = n.bit_length() - 1
    for i in range(n):
        j = _bit_reverse(i, bits)
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        w = _primitive_root_of_unity(length)
        if invert:
            w = _pow_int(w, _P - 2, _P)
        half = length // 2
        for i in range(0, n, length):
            wn = 1
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * wn % _P
                a[i + k] = (u + v) % _P
                a[i + k + half] = (u - v) % _P
                wn = wn * w % _P
        length <<= 1

    if invert:
        n_inv = _pow_int(n, _P - 2, _P)
        for i in range(n):
            a[i] = a[i] * n_inv % _P
    return a


def _next_power_of_two(n: int) -> int:
    """Smallest power of two >= n. Returns 1 for n <= 1."""
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def _modular(x: int) -> int:
    """Reduce x into [0, p). Handles negatives so callers can pass plain ints."""
    return x % _P


def _validate_int_input(x):
    if isinstance(x, bool) or not isinstance(x, (int,)):
        raise TypeError("input must be a list or tuple of ints")


def ntt(a):
    """Forward NTT of a list of ints, returned modulo 998244353.

    `a` must be non-empty and its length a power of two not exceeding 2**23.
    Inputs are reduced into [0, p); a TypeError is raised for non-int input
    and a ValueError for an empty list or an unsupported length.
    """
    if not isinstance(a, (list, tuple)):
        raise TypeError("input must be a list or tuple of ints")
    if len(a) == 0:
        raise ValueError("cannot transform an empty sequence")
    if len(a) & (len(a) - 1) != 0:
        raise ValueError("length must be a power of two")
    if len(a) > (1 << 23):
        raise ValueError("length exceeds the maximum supported transform size")
    for x in a:
        _validate_int_input(x)
    arr = [_modular(int(x)) for x in a]
    return _transform(arr, invert=False)


def intt(a):
    """Inverse of ntt. Same length constraints as ntt."""
    if not isinstance(a, (list, tuple)):
        raise TypeError("input must be a list or tuple of ints")
    if len(a) == 0:
        raise ValueError("cannot transform an empty sequence")
    if len(a) & (len(a) - 1) != 0:
        raise ValueError("length must be a power of two")
    if len(a) > (1 << 23):
        raise ValueError("length exceeds the maximum supported transform size")
    for x in a:
        _validate_int_input(x)
    arr = [_modular(int(x)) for x in a]
    return _transform(arr, invert=True)


def ntt_convolve(a, b):
    """Cyclic convolution of two integer sequences modulo 998244353.

    The result length equals the next power of two >= len(a)+len(b)-1,
    which is the smallest size that makes the cyclic convolution equal to
    the linear convolution. The caller is responsible for ensuring the true
    integer coefficients fit below p; see the README for the bound.
    """
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        raise TypeError("inputs must be lists or tuples of ints")
    if len(a) == 0 or len(b) == 0:
        raise ValueError("cannot convolve an empty sequence")
    target = _next_power_of_two(len(a) + len(b) - 1)
    if target > (1 << 23):
        raise ValueError("required transform size exceeds the supported maximum")
    for x in a:
        _validate_int_input(x)
    for x in b:
        _validate_int_input(x)
    A = [_modular(int(x)) for x in a] + [0] * (target - len(a))
    B = [_modular(int(x)) for x in b] + [0] * (target - len(b))
    fa = _transform(A, invert=False)
    fb = _transform(B, invert=False)
    fc = [fa[i] * fb[i] % _P for i in range(target)]
    return _transform(fc, invert=True)
