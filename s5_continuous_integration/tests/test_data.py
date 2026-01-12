from corrupted_mnist_pytest.data import corrupt_mnist
from tests import _PATH_DATA
import torch
import pytest
import os

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data folder not found")
def test_data():
    print("Testing data loading...")
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Training set should have 30000 samples, but got {}".format(len(train))
    assert len(test) == 5000, "Test set should have 5000 samples, but got {}".format(len(test))

    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
            
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()