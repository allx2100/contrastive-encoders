import torch
import dill
from models.vae import VAE
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

model = VAE(3, 32, 10, [8, 16, 32], kld_weight=1e-2)
loss_fn = model.loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

train_loss = []
recon_loss = []
kld = []
val_loss = []

print("Epoch\tTrain Loss\tRecon Loss\tKL Diverg\tVal Loss")
for i in range(300):
    train_vae(train_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld)
    validate(val_loader, model, loss_fn, val_loss)
    print(
        f"{i+1} ",
        f"\t{train_loss[-1]:>7f}",
        f"\t{recon_loss[-1]:>7f}",
        f"\t{kld[-1]:>7f}",
        f"\t{val_loss[-1]:>7f}",
    )
    # if (i + 1) % 20 == 0:
    #     save_data(model, f"saved_models/diag_{i+1}.pkl")