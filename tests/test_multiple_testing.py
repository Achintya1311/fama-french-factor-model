import pytest

from factors.multiple_testing import benjamini_hochberg, bonferroni, two_sided_normal_pvalue


def test_two_sided_normal_pvalue_matches_the_repos_own_1_96_convention():
    # Every `significant` flag elsewhere in this repo is exactly |t| > 1.96.
    # p < 0.05 here must agree with that cutoff, or this module would be
    # silently using a different definition of "significant".
    assert two_sided_normal_pvalue(1.96) == pytest.approx(0.05, abs=1e-3)
    assert two_sided_normal_pvalue(0.0) == pytest.approx(1.0)
    assert two_sided_normal_pvalue(-2.5) == two_sided_normal_pvalue(2.5)


def test_bonferroni_rejects_only_below_alpha_over_m():
    # threshold = 0.05 / 4 = 0.0125
    p_values = [0.001, 0.02, 0.03, 0.04]
    assert bonferroni(p_values, alpha=0.05) == [True, False, False, False]


def test_bonferroni_empty_input():
    assert bonferroni([], alpha=0.05) == []


def test_benjamini_hochberg_reproduces_the_1995_paper_worked_example():
    # Benjamini & Hochberg (1995), Table 1: m=15 p-values, alpha=0.05.
    # The paper's own BH procedure rejects exactly the four smallest.
    p_values = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
        0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
    ]
    reject = benjamini_hochberg(p_values, alpha=0.05)
    assert reject == [True, True, True, True] + [False] * 11


def test_benjamini_hochberg_is_never_more_conservative_than_bonferroni():
    # BH controls a weaker guarantee (FDR) than Bonferroni (FWER), so on any
    # input BH must reject at least everything Bonferroni does.
    p_values = [0.001, 0.004, 0.02, 0.03, 0.08, 0.2, 0.5, 0.9]
    bonf = bonferroni(p_values, alpha=0.05)
    bh = benjamini_hochberg(p_values, alpha=0.05)
    for is_bonf, is_bh in zip(bonf, bh):
        if is_bonf:
            assert is_bh


def test_benjamini_hochberg_rejects_nothing_when_nothing_qualifies():
    p_values = [0.2, 0.4, 0.6, 0.8]
    assert benjamini_hochberg(p_values, alpha=0.05) == [False, False, False, False]


def test_benjamini_hochberg_empty_input():
    assert benjamini_hochberg([], alpha=0.05) == []


def test_benjamini_hochberg_permutation_invariant_up_to_reordering():
    p_values = [0.001, 0.2, 0.03, 0.9, 0.04]
    reject = benjamini_hochberg(p_values, alpha=0.05)
    permuted = [p_values[i] for i in (4, 0, 3, 2, 1)]
    reject_permuted = benjamini_hochberg(permuted, alpha=0.05)
    expected = [reject[4], reject[0], reject[3], reject[2], reject[1]]
    assert reject_permuted == expected
