import numpy as np
import pytest

from finstats.timeseries.composite import simulate_ar1_garch11


def test_aligned_mean_and_variance_recursions():
    values, shocks, variances = simulate_ar1_garch11(.7, .1, .1, .8, 80, mu=2, rng=42)
    np.testing.assert_allclose(values[1:], 2 + .7*(values[:-1]-2) + shocks[1:])
    np.testing.assert_allclose(variances[1:], .1 + .1*shocks[:-1]**2 + .8*variances[:-1])
    assert values.shape == shocks.shape == variances.shape == (80,)
    assert np.all(variances > 0)
    repeated = simulate_ar1_garch11(.7, .1, .1, .8, 80, mu=2, rng=42)
    for actual, expected in zip((values, shocks, variances), repeated):
        np.testing.assert_array_equal(actual, expected)


def test_zero_ar_coefficient_leaves_garch_shocks():
    values, shocks, _ = simulate_ar1_garch11(0, .1, .1, .8, 30, mu=3, rng=3)
    np.testing.assert_allclose(values, 3+shocks)


def test_constant_variance_ar_population_moments():
    values, _, variances = simulate_ar1_garch11(.5, 1, 0, 0, 60000, mu=2, rng=17)
    assert abs(values.mean()-2) < .025
    assert abs(values.var()-1/(1-.5**2)) < .04
    np.testing.assert_array_equal(variances, np.ones(60000))


@pytest.mark.parametrize("arguments", [
    (1, .1, .1, .8, 10), (-1, .1, .1, .8, 10),
    (.5, .1, .2, .8, 10), (.5, 0, .1, .8, 10),
    (.5, .1, -.1, .8, 10), (.5, .1, .1, .8, 0),
    (.5, .1, .1, .8, True), (.5, .1, .1, .8, 1.5),
])
def test_invalid_parameters(arguments):
    with pytest.raises(ValueError):
        simulate_ar1_garch11(*arguments)


def test_nonfinite_population_mean():
    with pytest.raises(ValueError):
        simulate_ar1_garch11(.5, .1, .1, .8, 10, mu=np.nan)
