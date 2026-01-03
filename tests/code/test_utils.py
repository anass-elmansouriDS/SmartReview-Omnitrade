from scripts.utils import *
import pytest

@pytest.mark.parametrize(
    "input, expected",
    [
        ("True", True),
        ("t", True),
        ("False",False),
        ("f",False)
    ]
)
def test_str2bool(input, expected) :
    assert str2bool(input)==expected