import numpy as np
from PIL import ImageOps
from torchvision.datasets import MNIST

class CustomMNISTDataset(MNIST):
    """Custom MNIST Dataset inheriting torchvision.datasets.MNIST
       This class creates 3channels images of MNIIST.

       Args:
        root (str): MNIST dataset dir path
        train (bool): Train or Test
        transform (transforms.Compose): transforms
        target_transform: None
        download (bool): Allow to download or not
        invert (bool): If true, black white invert augmentation is applied
    """
    def __init__(self,
            root,
            train,
            transform,
            target_transform,
            download,
            invert,
        ):
        super().__init__(root, train, None, target_transform, download)
        self.custom_transform = transform
        self.invert = invert
    
    def __getitem__(self, idx):
        pil_image, label = super().__getitem__(idx)
        pil_image = pil_image.convert("RGB")
        if self.invert:
            if np.random.uniform() > 0.5:
                pil_image = ImageOps.invert(pil_image)
        if not self.custom_transform:
            return pil_image, label
        tensor = self.custom_transform(pil_image)
        return tensor, label

