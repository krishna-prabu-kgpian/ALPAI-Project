import pytest


@pytest.mark.xfail(reason="T-15 is implemented with Step 23")
def test_t15_frozen_config_guard():
    pytest.fail("T-15 placeholder")
