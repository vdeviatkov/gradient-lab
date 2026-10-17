import math

import pytest

from ml_scratch.pure.numerics import sigmoid, softplus


def test_shared_sigmoid_and_softplus_are_stable_at_saturation():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0
    assert softplus(1000.0) == 1000.0
    assert softplus(-1000.0) == 0.0
    assert softplus(0.0) == pytest.approx(math.log(2))
    assert softplus(-40.0) == pytest.approx(math.exp(-40), rel=1e-12)
