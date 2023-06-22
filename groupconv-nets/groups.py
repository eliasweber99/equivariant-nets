import torch


class C4:
    """
    rotations by n*90 degrees
    """
    size = 4
    indices = range(size)
    name = 'C4'
    
    @staticmethod
    def product(a: int, b: int) -> int:
        return (a + b)%4

    @staticmethod
    def inverse(index: int) -> int:
        return (-index)%4
    
    @staticmethod
    def action_on_grid(image: torch.Tensor, index: int) -> torch.Tensor:
        return image.rot90(index, dims=(-1,-2))
    
    @staticmethod
    def action_on_group(image: torch.Tensor, index: int) -> torch.Tensor:
        idx = [C4.product(index, j) for j in C4.indices]
        return C4.action_on_grid(image, index)[...,idx,:,:]



class D4:
    """
    rotations by n*90 degrees and reflection/mirroring
    """
    size = 8
    indices = range(size)
    name = 'D4'
    dict = {0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)}
    
    @staticmethod
    def product(a: int, b: int) -> int:
        m1, r1 = b//4, b%4
        m2, r2 = a//4, a%4
        m = (m1 + m2)%2
        r = ((-1)**(m1-m2)*r1 + (-1)**(m1+m2)*r2)%4
        return 4*m + r

    @staticmethod
    def inverse(index: int) -> int:
        m1, r1 = index//4, index%4
        m = (-m1)%2
        r = (-r1)%4
        return D4.product(r, 4*m)
    
    @staticmethod
    def action_on_grid(image: torch.Tensor, index: int) -> torch.Tensor:
        m, r = index//4, index%4
        if m == 1: image = image.flip(dims=(-1,))
        return image.rot90(r, dims=(-1,-2))
    
    @staticmethod
    def action_on_group(image: torch.Tensor, index: int) -> torch.Tensor:
        idx = [D4.product(index, j) for j in D4.indices]
        return D4.action_on_grid(image, index)[...,idx,:,:]