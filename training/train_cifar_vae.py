import torch
import dill
from models.cifar_vae import VAE
from data.utils import *
from training.utils import *

with open("data/cifar_data_loaders.pkl", "rb") as f:
    data_loaders = dill.load(f)

train_loader = data_loaders["train"]
val_loader = data_loaders["val"]
test_loader = data_loaders["test"]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

model = VAE(3, 32, 128, [32, 64, 128, 256], kld_weight=1e-3)
loss_fn = model.loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loss = []
recon_loss = []
kld = []
val_loss = []

print("Epoch\tTrain Loss\tRecon Loss\tKL Diverg\tVal Loss")
for i in range(2000):
    train_vae(train_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld, device)
    validate(val_loader, model, loss_fn, val_loss, device)
    print(
        f"{i+1} ",
        f"\t{train_loss[-1]:>7f}",
        f"\t{recon_loss[-1]:>7f}",
        f"\t{kld[-1]:>7f}",
        f"\t{val_loss[-1]:>7f}",
    )
    if (i + 1) % 100 == 0:
        save_data(model, f"saved_models/cifar_vae_{i+1}.pkl")