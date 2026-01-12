import torch
#import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from dotenv import load_dotenv
import os

load_dotenv()
wandb.login()
project = os.getenv("WANDB_PROJECT")
entity = os.getenv("WANDB_ENTITY")
#app = typer.Typer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(version_base=None, config_path="config", config_name="config")
# @app.command()  # Literally this decorator convers the function into a CLI command, where the arguments can be passed via command line using --argname value
def train(cfg: DictConfig) -> None:
    """Train a model on MNIST."""
    print("Training params:")
    print(OmegaConf.to_yaml(cfg))
    #print(f"Learning rate: {cfg.optimizer.lr}")
    print(f"Epochs: {cfg.hyperparams.epochs}")
    print(f"Batch size: {cfg.hyperparams.batch_size}")

    run = wandb.init(project=project, entity=entity, job_type="training", config=OmegaConf.to_container(cfg), name=f"run_{wandb.util.generate_id()}_{cfg.optimizer._target_.split('.')[-1]}_lr{cfg.optimizer.lr}_bs{cfg.hyperparams.batch_size}")

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.hyperparams.batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparams.lr)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    print(f"Using optimizer: {optimizer}")

    train_statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(cfg.hyperparams.epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()  # Batch accuracy
            epoch_loss += loss.item()
            epoch_accuracy += accuracy

        train_statistics["train_loss"].append(epoch_loss / len(train_loader))
        train_statistics["train_accuracy"].append(epoch_accuracy / len(train_loader))
        print(
            f"Epoch [{epoch + 1}/{cfg.hyperparams.epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy / len(train_loader):.4f}"
        )
        wandb.log({"train_loss": train_statistics["train_loss"][-1], "train_accuracy": train_statistics["train_accuracy"][-1]})

    # Plot and save model
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved as trained_model.pth")

    artifact = wandb.Artifact("my_awesome_model", 
                              type="model",
                              description="A trained MNIST model",
                              metadata={"epochs": cfg.hyperparams.epochs,
                                        "batch_size": cfg.hyperparams.batch_size,
                                        "optimizer": cfg.optimizer._target_,
                                        "learning_rate": cfg.optimizer.lr,
                                        "final_accuracy": sum(train_statistics["train_accuracy"]) / len(train_statistics["train_accuracy"])})
    artifact.add_file("trained_model.pth")
    run.log_artifact(artifact)
    run.link_artifact(
        artifact=artifact,
        target_path="wandb-registry-corrupt_mnist_models/corrupt_mnist_models",
        aliases=["latest"]
    )

    # Plot training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_statistics["train_loss"], label="Train Loss")
    axs[1].plot(train_statistics["train_accuracy"], label="Train Accuracy", color="orange")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Value")
    axs[0].set_title("Training Loss")
    axs[0].legend()
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Value")
    axs[1].set_title("Training Accuracy")
    axs[1].legend()
    plt.savefig("training_statistics.png")
    # save plot to wandb
    wandb.log({"training_statistics": wandb.Image("training_statistics.png")})

def evaluate(model_checkpoint: str = "trained_model.pth", batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    _, test_set = corrupt_mnist()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            batch_accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
            accuracy += batch_accuracy

    accuracy /= len(test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")

    def visualize():
        """Visualize some predictions."""
        print("Visualizing predictions...")
        model.eval()
        images, labels = next(iter(test_loader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(images)
        outputs = torch.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1)

        # Visualize first 5 images, their predicted and true labels
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            axs[i].imshow(images[i].cpu().squeeze(), cmap="gray")
            axs[i].set_title(f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}")
            axs[i].axis("off")
            plt.savefig("predictions.png")

    visualize()


if __name__ == "__main__":
    train()
