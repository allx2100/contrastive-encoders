import torch
import dill
from models.swae import SWAE
from data.utils import *
from training.utils import *


def main():
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

    model = SWAE(3, 32, 10, [8, 16, 32], swd_weight=1.0, n_proj=50)
    loss_fn = model.loss_function
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    train_loss = []
    recon_loss = []
    swd = []
    val_loss = []

    print("Epoch\tTrain Loss\tRecon Loss\tSWD Loss\tVal Loss")
    for i in range(2):
        train_swae(train_loader, model, loss_fn,
                   optimizer, train_loss, recon_loss, swd)
        validate(val_loader, model, loss_fn, val_loss)
        print(
            f"{i+1} ",
            f"\t{train_loss[-1]:>7f}",
            f"\t{recon_loss[-1]:>7f}",
            f"\t{swd[-1]:>7f}",
            f"\t{val_loss[-1]:>7f}",
        )
        # if (i + 1) % 20 == 0:
        #     save_data(model, f"saved_models/diag_{i+1}.pkl")


if __name__ == "__main__":
    main()
