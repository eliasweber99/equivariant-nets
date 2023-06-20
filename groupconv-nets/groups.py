import torch
from typing import Tuple


class C4:
    """
    rotations by n*90 degrees
    """
    size = 4
    indices = range(size)
    
    @staticmethod
    def product(a: int, b: int) -> int:
        return (a + b)%4

    @staticmethod
    def inverse(index: int) -> int:
        return (-index)%4
    
    @staticmethod
    def action(image: torch.tensor, index: int) -> torch.tensor:
        return image.rot90(index, dims=(-1,-2))
    
    @staticmethod
    def representation(image: torch.tensor, index: int) -> torch.tensor:
        """
        name is wip
        """
        idx = [C4.product(j, index) for j in C4.indices]
        return C4.action(image, index)[...,idx,:,:]



class D4:
    """
    rotations by n*90 degrees and reflection/mirroring
    """
    size = 8
    indices = range(size)
    # 0: 0,0; 1: 0,1; ...; 7: 1,3
    
    @staticmethod
    def product(a: int, b: int) -> int:
        m1, r1 = a//4, a%4
        m2, r2 = b//4, b%4
        m = (m1 + m2)%2
        r = (r1 + (-1)**m1 * r2)%4
        return 2*m + r

    @staticmethod
    def inverse(index: int) -> int:
        m1, r1 = index//4, index%4
        m = (-m1)%2
        r = (-r1)%4
        return 2*m + r
    
    @staticmethod
    def action(image: torch.tensor, index: int) -> torch.tensor:
        m, r = index//4, index%4
        if m == 1: image = image.flip(dims=(-2,))
        return image.rot90(r, dims=(-1,-2))
    
    @staticmethod
    def representation(image: torch.tensor, index: int) -> torch.tensor:
        idx = [D4.product(j, index) for j in D4.indices]
        return D4.action(image, index)[...,idx,:,:]