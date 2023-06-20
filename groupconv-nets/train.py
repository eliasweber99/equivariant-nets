import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm as tqdm

from typing import Callable, Optional, Tuple, Union
import argparse

def train(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    max_epochs: int, 
    lr: float,
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    # continue_from: Optional[Tuple],
    *args,
    reg_str: float=.0, 
    reg_ord: int=1,
    tqdm = tqdm_n,
    **kwargs
    ):
    """
    Args:
        - model: The model which to train
        - optimizer: The desired optimizer
        - criterion: The loss criterion to optimize
        - max_epochs: Maximum number of training epochs
        - train_loader: Dataloader for training
        - val_loader: DataLoader for validation
        - reg_str(float): Regularization strength
        - reg_ord(int): Regularization order (L1 / L2)
        - device: 
    """
    train_losses = []
    val_losses = []
    reg_losses = []
    train_acc = []
    val_acc = []
    
    optimizer = optimizer(model.parameters(), lr=lr)
    device = model.device
    
    pbar = tqdm(range(max_epochs))
    pbar.set_description('00.00%')
    for e in pbar:
        train_bar = tqdm(total=len(train_loader), leave=False)
        val_bar = tqdm(total=len(val_loader), leave=False)
        for x, y in train_loader:
            model.zero_grad()
            
            x, y = x.to(device), y.to(device)

            probs = model(x)

            crit_loss = criterion(input=probs, target=y)
            reg_loss = reg_str*model.regularize(reg_ord)

            loss = crit_loss + reg_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_losses.append(crit_loss.item())
                reg_losses.append(reg_loss.item())
                preds = torch.argmax(probs, dim=-1)
                train_acc.append((preds==y).float().mean())
            train_bar.update()
        
        epoch_loss = 0
        epoch_acc = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                probs = model(x)
                epoch_loss += criterion(probs, y)
                preds = torch.argmax(probs, dim=-1)
                epoch_acc += (preds==y).float().mean()
            val_bar.update()
        train_bar.close()
        val_bar.close()
        val_losses.append(epoch_loss/len(val_loader))
        val_acc.append(epoch_acc/len(val_loader))
        pbar.set_description(f'{100*val_acc[-1]:2.2f}%')
    
    return (torch.tensor(train_losses).cpu(), 
            torch.tensor(reg_losses).cpu(),
            torch.tensor(train_acc).cpu(), 
            torch.tensor(val_losses).cpu(), 
            torch.tensor(val_acc).cpu()
           )