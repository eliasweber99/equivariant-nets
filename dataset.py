import torch
from torchvision.datasets import MNIST
from torchvision import transforms as T

from typing import Callable, Optional
from utils import vertical_shift, horizontal_shift

from PIL import Image



class customMNIST(MNIST):
    def __init__(self, 
                 root: str = './data/', 
                 train: bool = True,
                 transform: Optional[Callable] =\
                     T.Compose([T.ToTensor(), 
                                T.Normalize(.5, .5)
                               ]),
                 target_transform:Optional[Callable] = None, 
                 vshift: int=0, 
                 hshift: int=0
                ):
        super().__init__(root, train, transform, target_transform, download=True)
        
        self.vshift = vshift
        self.hshift = hshift
        
        self.data = vertical_shift(horizontal_shift(self.data, self.hshift), self.vshift)
    
    def __getitem__(self, index):
        """
        copied from torchvision docs
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target