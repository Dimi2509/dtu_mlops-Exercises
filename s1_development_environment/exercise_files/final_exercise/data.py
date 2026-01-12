import torch

DATA_PATH = "C:\\Users\\ivand\\OneDrive\\Uni Studies\\DTU - Masters\\Spring 2026\\02476 - MLOps\\dtu_mlops\\s1_development_environment\\exercise_files\\final_exercise\\data\\corruptmnist\\corruptmnist_v1"
n_training_pt_files = 6


def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    # Use list comprehension to load .pt files with the data(concatenate) using torch.load
    train_images = []
    test_images = []
    train_targets = []
    test_targets = []

    for i in range(n_training_pt_files):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_targets.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))

    train_images = torch.cat(
        train_images
    )  # Concatenate the list of tensors into a single tensor, from 5000x28x28 for each .pt to 30000x28x28
    train_targets = torch.cat(train_targets)
    # print(f" Train images: {train_images.shape}")
    # print(f" Train targets: {train_targets.shape}")

    test_images = torch.load(f"{DATA_PATH}/test_images.pt")
    test_targets = torch.load(f"{DATA_PATH}/test_target.pt")
    # print(f" Test images: {test_images.shape}")
    # print(f" Test targets: {test_targets.shape}")

    train_images = train_images.unsqueeze(
        1
    ).float()  # Add channel dimension and convert to float32, from 30000x28x28 to 30000x1x28x28
    test_images = test_images.unsqueeze(
        1
    ).float()  # Add channel dimension and convert to float32, from 5000x28x28 to 5000x1x28x28
    train = torch.utils.data.TensorDataset(train_images, train_targets)
    test = torch.utils.data.TensorDataset(test_images, test_targets)

    # print(f" Final Train images: {train_images.shape}, dtype: {train_images.dtype}")
    # print(f" Final Test images: {test_images.shape}, dtype: {test_images.dtype}")

    return train, test


if __name__ == "__main__":
    corrupt_mnist()
