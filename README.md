# number_theoretic_transform

Exact integer convolution via the Number Theoretic Transform over the fixed prime p = 998244353.

```python
from number_theoretic_transform import ntt, intt, ntt_convolve

assert intt(ntt([1, 2, 3, 4])) == [1, 2, 3, 4]
assert ntt_convolve([1, 1], [1, 1]) == [1, 2, 1, 0]
```

## Why this exists

Floating-point FFT gives approximate convolution; for integer sequences whose true coefficients are known to be small, an NTT over a suitably large prime recovers them exactly with no rounding error. This library implements that for a single fixed prime (998244353) that admits transform lengths up to 2**23. That covers any practical problem where the integer coefficients stay below roughly 2**30 * max(input)^2 * length — enough for typical competitive-programming and small-bignum workloads.

## What you need to know before using it

- Inputs to `ntt`/`intt` must be a non-empty list or tuple whose length is a power of two and at most 2**23. `ntt_convolve` pads internally to the next power of two and returns that length; callers should trim trailing zeros if they only want the linear-convolution prefix.
- The prime is fixed. If your true integer coefficients can exceed 998244353 you will get silent wraparound — that is the trade-off for a single-modulus library, not a bug. For unbounded convolution you would need a multi-prime (CRT) scheme, which this library deliberately does not provide.
- Negative inputs are reduced into [0, p), so `intt(ntt([-1]))` yields `[p - 1]`, not `-1`. Reduce on your side if you need signed output.

## Exported names

- `ntt(a)` — forward transform, returns a list of residues mod 998244353.
- `intt(a)` — inverse transform of a residue list.
- `ntt_convolve(a, b)` — cyclic convolution of two integer sequences, zero-padded to the next power of two >= len(a)+len(b)-1 so the cyclic and linear convolutions coincide.
