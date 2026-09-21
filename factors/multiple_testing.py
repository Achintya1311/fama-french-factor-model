"""Multiple-testing corrections (Day 7).

Every alpha this repo has reported so far - 30 industry/model rows in
``decay_report``, 9 ticker/model rows in ``screen_check`` - was one test
among many run against the same data. The README's own limitations section
already says so in prose ("testing several specifications on one dataset is
multiple testing"); this module is that prose turned into a number: how many
of the raw "significant at 5%" flags survive once the fact that dozens of
tests were run is actually accounted for.

Two standard corrections, both operating on a flat list of p-values from one
"family" of tests (tests that should be corrected together, e.g. all 30
decay_report rows under one standard-error convention):

- Bonferroni: reject only if p < alpha/m. Controls the family-wise error
  rate (probability of *any* false positive) - conservative, appropriate
  when even one false "the alpha survived" claim would be costly.
- Benjamini-Hochberg: a step-up procedure controlling the expected
  proportion of false positives among the rejections (the false discovery
  rate) rather than the chance of any false positive at all - less
  conservative, the more common choice when scanning many specifications
  for candidates worth a closer look.
"""

from __future__ import annotations

from scipy.stats import norm


def two_sided_normal_pvalue(t_stat: float) -> float:
    """Two-sided p-value from a t-stat, via the normal approximation.

    Every ``significant``/``significant_nw``/``significant_ols`` flag
    already computed elsewhere in this repo (``capm.py``, ``multifactor.py``,
    ``diagnostics.py``) is exactly ``|t| > 1.96`` - the normal, not exact-t,
    critical value, applied regardless of degrees of freedom. Using the same
    normal approximation here keeps ``p_value < 0.05`` identical to those
    existing flags rather than quietly introducing a second, different
    definition of "significant" for this module alone.
    """
    return float(2.0 * norm.sf(abs(t_stat)))


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Reject flags after a Bonferroni family-wise-error correction.

    Rejects test ``i`` iff ``p_values[i] < alpha / len(p_values)``. An empty
    input returns an empty list rather than raising.
    """
    m = len(p_values)
    if m == 0:
        return []
    threshold = alpha / m
    return [p < threshold for p in p_values]


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Reject flags after Benjamini-Hochberg false-discovery-rate control.

    Standard BH step-up procedure: sort p-values ascending, find the largest
    rank ``k`` with ``p_(k) <= (k / m) * alpha``, and reject that test and
    every test with a smaller p-value. Returns rejection flags in the
    original input order. An empty input returns an empty list.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]

    threshold_rank = 0  # 0 means "reject nothing"
    for k in range(m, 0, -1):
        if sorted_p[k - 1] <= (k / m) * alpha:
            threshold_rank = k
            break

    reject_sorted = [rank < threshold_rank for rank in range(m)]
    reject = [False] * m
    for rank, original_index in enumerate(order):
        reject[original_index] = reject_sorted[rank]
    return reject
