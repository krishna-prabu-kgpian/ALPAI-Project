import pytest


@pytest.mark.xfail(reason="T-12 is implemented with Step 19")
def test_t12_cvar_slicing():
    pytest.fail("T-12 placeholder")
