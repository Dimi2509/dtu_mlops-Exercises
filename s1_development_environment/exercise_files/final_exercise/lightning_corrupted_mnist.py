import torch
from torch import nn
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from data import corrupt_mnist
import os
from dotenv import load_dotenv

load_dotenv()
wandb.login()
project = os.getenv("WANDB_PROJECT")
entity = os.getenv("WANDB_ENTITY")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyAwesomeModel(pl.LightningModule):
    """MNIST dataset, create a CNN. Input image is 1x28x28"""

    def __init__(self, k=10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)  # 32x26x26
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # 64x24x24
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)  # 128x22x22
        self.fc1 = nn.Linear(in_features=128 * 1 * 1, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=k)
        self.dropout = nn.Dropout(p=0.3)

        # Metrics
        self.train_shared_metrics = []
        self.val_shared_metrics = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 32x13x13
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 64x6x6
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 128x3x3
        # print(f"x.shape before flatten: {x.shape}")
        x = torch.flatten(
            x, start_dim=1
        )  # Inpput is 1x128x1x1 (batch size 1, 128 channels, 1x1 image), flattening from 1 outputs 1x128
        # print(f"x.shape after flatten: {x.shape}")
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.train_shared_metrics.append({'loss': loss, 'acc': acc})
        # self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        # self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        epoch_loss = torch.stack([l['loss'] for l in self.train_shared_metrics]).mean()
        epoch_acc = torch.stack([a['acc'] for a in self.train_shared_metrics]).mean()
        self.log("train_acc", epoch_acc, on_step=False, on_epoch=True)
        self.log("train_loss", epoch_loss, on_step=False, on_epoch=True)
        self.train_shared_metrics.clear()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.val_shared_metrics.append({'loss': loss, 'acc': acc})
        # self.log("val_acc", acc, on_step=False, on_epoch=True)
        # self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        epoch_loss = torch.stack([l['loss'] for l in self.val_shared_metrics]).mean()
        epoch_acc = torch.stack([a['acc'] for a in self.val_shared_metrics]).mean()
        self.log("val_acc", epoch_acc, on_step=False, on_epoch=True)
        self.log("val_loss", epoch_loss, on_step=False, on_epoch=True)
        self.val_shared_metrics.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    model = MyAwesomeModel()
    train_dataset, test_dataset = corrupt_mnist()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, persistent_workers=True)

    callbacks = [
        ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/', filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'),
        EarlyStopping(monitor='val_loss', patience=3, mode='min')
    ]
     
    
    trainer = pl.Trainer(callbacks=callbacks, max_epochs=10, accelerator="cuda", logger=WandbLogger(name=f"run_{wandb.util.generate_id()}_pl", project=project, entity=entity), log_every_n_steps=len(train_loader))
    trainer.fit(model, train_loader, test_loader)

