"""
Uncertainty for small-sample proportions.

Most figures in this project come from small sets -- sixteen gold questions,
sixteen unanswerable queries, a few dozen probes -- and a proportion from n=16
moves by 0.0625 per item. A 2026-10 peer review noted that headline figures
were reported as bare point estimates (and that 0.966 -> 1.000 is one removed
citation out of 29). Every proportion the paper reports from a sample of this
size is now printed with a Wilson score interval, which, unlike the normal
approximation, behaves sensibly at 0/n and n/n -- exactly the perfect scores
this project most often reports, and the ones most in need of a bound.
"""
from math import sqrt


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt(k: int, n: int) -> str:
    """'0.938 (15/16, 95% CI 0.72-0.99)'"""
    lo, hi = wilson(k, n)
    p = k / n if n else float("nan")
    return f"{p:.3f} ({k}/{n}, 95% CI {lo:.2f}-{hi:.2f})"
