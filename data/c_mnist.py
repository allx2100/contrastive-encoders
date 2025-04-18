from torch.utils.data import Dataset

class ContrastiveMNIST(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2, y

    def __len__(self):
        return len(self.base_dataset)
