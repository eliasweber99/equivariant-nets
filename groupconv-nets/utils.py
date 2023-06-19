import torch
import matplotlib.pyplot as plt

def vertical_shift(img: torch.Tensor, d: int = 1):
    """
    Translates img by d along vertical axis
    Args:
        - img (tensor): The image to be shifted
        - d (int): The distance by which to shift
    Returns:
        - (Tensor) a shifted copy of the input image
    """
    if d == 0:
        return img.detach().clone()
    shifted = torch.empty(img.shape, dtype=img.dtype)
    shifted[...,:-d,:] = img[...,d:,:].detach().clone()
    shifted[...,-d:,:] = img[...,:d,:].detach().clone()
    return shifted

def horizontal_shift(img: torch.Tensor, d: int = 1):
    """
    input images must have spatial dimensions as last two dimensions
    """
    if d == 0:
        return img.detach().clone()
    shifted = torch.empty(img.shape, dtype=img.dtype)
    shifted[...,:-d] = img[...,d:].detach().clone()
    shifted[...,-d:] = img[...,:d].detach().clone()
    return shifted

def plot_losses(trl, rel, val, tra, vaa):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0][0].plot(trl)
    ax[0][0].plot(rel)
    ax[0][0].set_xlabel('iteration')
    ax[0][0].set_ylabel(r'$\log\mathcal{L}$')
    ax[0][0].set_yscale('log')
    
    ax[0][1].plot(val)
    ax[0][1].set_xlabel('iteration')
    ax[0][1].set_ylabel(r'$\log\mathcal{L}$')
    ax[0][1].set_yscale('log')
    
    ax[1][0].plot(tra)
    ax[1][0].set_xlabel('epoch')
    ax[1][0].set_ylabel('prediction accuracy')
    
    ax[1][1].plot(vaa)
    ax[1][1].set_xlabel('epoch')
    ax[1][1].set_ylabel('prediction accuracy')
    
    return fig, ax

def calc_accuracy(model, loader):
    with torch.no_grad():
        accuracies = []
        for x, y in loader:
            x, y = x.to(model.device), y.to(model.device)
            probs = model(x)
            preds = probs.argmax(-1)
            accuracies.append((preds==y).float().mean())
    return torch.tensor(accuracies).mean()