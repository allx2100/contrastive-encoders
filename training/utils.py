import torch

def train_vae(data_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld_list, device):
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    running_loss = 0.0
    running_recon = 0.0
    running_kld = 0.0

    model.train()
    model.to(device)
    for i, data in enumerate(data_loader):
        inputs, _ = data
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        total_loss = loss_fn(outputs)
        loss = total_loss[0]
        r_loss = total_loss[1]
        KLD = total_loss[2]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_recon += r_loss.item()
        running_kld += KLD.item()

    train_loss.append(running_loss / size * batch_size)
    recon_loss.append(running_recon / size * batch_size)
    kld_list.append(running_kld / size * batch_size)

    return running_loss / size

def train_cvae(data_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld_list, con_losses, device):
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    running_loss = 0.0
    running_recon = 0.0
    running_kld = 0.0
    running_con_loss = 0.0

    model.train()
    model.to(device)

    for i, data in enumerate(data_loader):
        x1, x2, _ = data 
        x1, x2 = x1.to(device), x2.to(device)

        optimizer.zero_grad()

        outputs = model(x1, x2)

        total_loss = loss_fn(outputs, contrastive=True)
        loss, r_loss, KLD, c_loss = total_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_recon += r_loss.item()
        running_kld += KLD.item()
        running_con_loss += c_loss.item()

    train_loss.append(running_loss / size * batch_size)
    recon_loss.append(running_recon / size * batch_size)
    kld_list.append(running_kld / size * batch_size)
    con_losses.append(running_con_loss / size * batch_size)

    return running_loss / size

def train_cvae_mnist(data_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld_list, con_losses, device):
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    running_loss = 0.0
    running_recon = 0.0
    running_kld = 0.0
    running_con_loss = 0.0

    model.train()
    model.to(device)

    for i, data in enumerate(data_loader):
        x1, x2 = data
        x1, x2 = x1.to(device), x2.to(device)

        optimizer.zero_grad()

        outputs = model(x1, x2)

        total_loss = loss_fn(outputs, contrastive=True)
        loss, r_loss, KLD, c_loss = total_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_recon += r_loss.item()
        running_kld += KLD.item()
        running_con_loss += c_loss.item()

    train_loss.append(running_loss / size * batch_size)
    recon_loss.append(running_recon / size * batch_size)
    kld_list.append(running_kld / size * batch_size)
    con_losses.append(running_con_loss / size * batch_size)

    return running_loss / size

def train_swae(data_loader, model, loss_fn, optimizer, train_loss, recon_loss, swd_list):
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    running_loss = 0.0
    running_recon = 0.0
    running_swd = 0.0

    model.train()
    for i, data in enumerate(data_loader):
        inputs, _ = data

        optimizer.zero_grad()
        outputs = model(inputs)
        x_recon, z = outputs

        total_loss = loss_fn(x_recon, inputs, z)
        loss = total_loss[0]
        r_loss = total_loss[1]
        SWD = total_loss[2]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()
        running_recon += r_loss.item()
        running_swd += SWD.item()

    train_loss.append(running_loss / size * batch_size)
    recon_loss.append(running_recon / size * batch_size)
    swd_list.append(running_swd / size * batch_size)

    return running_loss / size


def validate(data_loader, model, loss_fn, val_loss, device):
    running_vloss = 0
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size

    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, vdata in enumerate(data_loader):
            vinputs, _ = vdata
            vinputs = vinputs.to(device)
            voutputs = model(vinputs)
            if model.__class__.__name__ == "SWAE":
                x_recon, z = voutputs
                vloss = loss_fn(x_recon, vinputs, z)[0]
            else:
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs)[0]
            running_vloss += vloss.item()

    val_loss.append(running_vloss / size * batch_size)
    return running_vloss / size * batch_size

def validate_cvae(data_loader, model, loss_fn, val_loss, device):
    running_vloss = 0.0
    running_recon = 0.0
    running_kld = 0.0
    running_con = 0.0
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size

    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            x1, x2, _ = data 
            x1, x2 = x1.to(device), x2.to(device)

            outputs = model(x1, x2)
            total_loss, recon_loss, kld, con_loss = loss_fn(outputs)

            running_vloss += total_loss.item()
            running_recon += recon_loss.item()
            running_kld += kld.item()
            running_con += con_loss.item()

    val_loss.append(running_vloss / size * batch_size)
    return running_vloss / size * batch_size

def validate_cvae_mnist(data_loader, model, loss_fn, val_loss, device):
    running_vloss = 0.0
    running_recon = 0.0
    running_kld = 0.0
    running_con = 0.0
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size

    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            x1, x2 = data
            x1, x2 = x1.to(device), x2.to(device)

            outputs = model(x1, x2)
            total_loss, recon_loss, kld, con_loss = loss_fn(outputs)

            running_vloss += total_loss.item()
            running_recon += recon_loss.item()
            running_kld += kld.item()
            running_con += con_loss.item()

    val_loss.append(running_vloss / size * batch_size)
    return running_vloss / size * batch_size



def test(data_loader, model, device):
    model.eval()
    model.to(device)
    preds = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.append(outputs)
    return preds
