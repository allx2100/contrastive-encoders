import torch
import dill
from models.contrastive_cifar_vae import CVAE
from data.utils import *
from training.utils import *

with open("data/contrastive_cifar_data_loaders.pkl", "rb") as f:
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

model = CVAE(3, 32, 128, [32, 64, 128, 256], kld_weight=1e-2)
loss_fn = model.loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loss = []
recon_loss = []
kld = []
val_loss = []
con_loss = []

print("Epoch\tTrain Loss\tRecon Loss\tKL Diverg\tCon Loss\tVal Loss")
for i in range(2000):
    train_cvae(train_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld, con_loss, device)
    validate_cvae(val_loader, model, loss_fn, val_loss, device)
    print(
        f"{i+1} ",
        f"\t{train_loss[-1]:>7f}",
        f"\t{recon_loss[-1]:>7f}",
        f"\t{kld[-1]:>7f}",
        f"\t{con_loss[-1]:>7f}"
        f"\t{val_loss[-1]:>7f}",
    )
    if (i + 1) % 100 == 0:
        save_data(model, f"saved_models/cifar_cvae_{i+1}.pkl")