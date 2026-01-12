from corrupted_mnist_pytest.model import MyAwesomeModel
import torch
import pytest

def test_model():
    model = MyAwesomeModel()
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    assert output.shape == (2, 10), "Output shape should be (batch_size, num_classes)"

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)