import pytest
from scripts.predict import *

def test_parse_args_predict() :
    args=parse_args_predict()
    assert args is not None