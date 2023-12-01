import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.mnist_dataset import CustomMNISTDataset
from model import SimpleCNN

def create_dataset(is_train: bool = True, normalize: bool = True, invert: bool = True):
    """Create MNIST data set
       Dataset path is "./data"
       If not exists, download will be started automatically.

    Args:
        is_train (bool): Train or test
        normalize (bool): Normalize or not
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])
    if normalize:
        return CustomMNISTDataset(
            root="./data",
            train=is_train,
            transform=transform,
            target_transform=None,
            download=True,
            invert=invert
        )
    return CustomMNISTDataset(
        root="./data",
        train=is_train,
        transform=None,
        target_transform=None,
        download=True,
        invert=False
    )


def create_dataloader(is_train: bool = True, batch_size: int = 512, shuffle: bool = True):
    """Create MNIST data loader

    Args:.
        is_train (bool): Train or test
        batch_size (int): batch size
        shuffle (bool): Shuffle flag
    """
    dataset = create_dataset(is_train=is_train, normalize=True, invert=is_train)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader

def prepare_model(pth_model_path = None):
    model = SimpleCNN(3, 32, 10)
    if pth_model_path:
        print("Load pretrained model weights")
        checkpoint = torch.load(pth_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Init model weights")
        model.apply(init_weights)
    return model

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)