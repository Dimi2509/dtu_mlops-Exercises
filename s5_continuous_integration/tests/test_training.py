import torch
from corrupted_mnist_pytest.model import MyAwesomeModel
import pytest
import re

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))

    # Regex was using [] which have special meaning, so use re.escape to avoid assestion error
    with pytest.raises(ValueError, match=re.escape('Expected each sample to have shape [1, 28, 28]')): 
        model(torch.randn(1,1,28,29))

def test_dropout_eval_mode():
    model = MyAwesomeModel()
    model.eval()
    assert not model.dropout.training, "Dropout should be in eval mode when model is in eval mode"