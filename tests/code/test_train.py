import pytest
from scripts.train import *

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full training tests require a GPU."
)
def test_load_model() :
    pass

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full training tests require a GPU."
)
def test_setup_lora_adapters() :
    pass

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full training tests require a GPU."
)
def test_setup_training_args() :
    pass

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full training tests require a GPU."
)
def test_setup_training() :
    pass


