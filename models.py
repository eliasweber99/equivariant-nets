import torch
from torch import nn
from typing import Any, List, Tuple, Union



class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int, 
        hidden_dim: List[int], 
        dropout: float=0.0, 
        sigmoid_out: bool=False,
        device: Union[torch.device, str] = torch.device('cpu')
    ):
        """
        Classic Multilayer Perceptron architecture.
        Args:
            - in_dim (int): input dimension
            - hidden_dim (list of int): hidden dimensions
            - dropout (float): dropout probability
            - device (torch.device or string): device
        """
        super().__init__()
        # input layer
        self.linear0 = nn.Sequential(
            nn.Flatten(1), 
            nn.Dropout(p=dropout), 
            nn.Linear(in_dim, hidden_dim[0]), 
            nn.ReLU()
        )
        # hidden layers
        for k, (i, o) in enumerate(zip(hidden_dim[:-1], hidden_dim[1:])):
            self.add_module(
                f'linear{k+1}', 
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(i, o),
                    nn.ReLU() if k+1==len(hidden_dim) else (nn.Sigmoid() if sigmoid_out else nn.Identity())
                )
            )
        
        # other stuff
        self.n_layers=len(hidden_dim)
        self.n_params = sum([p.numel() for p in self.parameters()])
        
        self.device = device
        self.to(device)

    def forward(self, x: torch.tensor):
        """
        forward pass
        Args:
            - x (Tensor): batched or unbatched image input with channel dimension
        returns: 
            - logits
        """
        for i in range(self.n_layers):
            x = self.get_submodule(f'linear{i}')(x)
        return x.squeeze()
    
    def regularize(self, p: int = 1):
        """
        Computes p-norm over network parameters normalized to the number of parameters
        Args:
            - p (int): Order of matrix norm
        Returns:
            - p-norm of network parameters
        """
        norms = []
        for param in self.parameters():
            norms.append((torch.abs(param)**p).sum())
        return sum(norms)**(1/p)/self.n_params






class CNN(nn.Module):
    def __init__(
        self, 
        channels: List, 
        kernel_sizes: Union[List[int], int], 
        paddings: Union[List[int], int], 
        output_dims: Tuple[int, int], 
        dropout: float = 0.0, 
        sigmoid_out: bool=False,
        device: Union[torch.device, str] = torch.device('cpu')
    ):
        """
        CNN model
        Args:
            - channels (list): list of number of channels through layers
            - kernel_sizes (list of int or int): list of kernel sizes
                through layers, may be int if same for all layers
            - paddings (list of int or int): list of paddings through
                layers, may be int if same for all layers
            - output_dims (tuple of int): in and output sizes of linear
                output layer
            - dropout (float): dropout probability
            - device (torch.device or string): device
        """
        super().__init__()
        if type(kernel_sizes) is int:
            kernel_sizes = [kernel_sizes] * (len(channels)-1)
        if type(paddings) is int:
            paddings = [paddings] * (len(channels)-1)
        
        # conv layers
        for j, (ci, co, k, p) in enumerate(zip(
            channels[:-1],
            channels[1:], 
            kernel_sizes, 
            paddings)
        ):
            self.add_module(
                f'conv{j}', 
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Conv2d(ci, co, kernel_size=k, padding=p),
                    nn.MaxPool2d(2),
                    nn.ReLU()
                )
            )
        # lin output layer
        self.lin_layer = nn.Sequential(
            nn.Flatten(start_dim=1), 
            nn.Linear(*output_dims),
            nn.Sigmoid() if sigmoid_out else nn.Identity()
        )
        
        self.n_conv_layers = len(kernel_sizes)
        self.n_params = sum([p.numel() for p in self.parameters()])
        
        self.device = device
        self.to(device)

    def forward(self, x):
        """
        forward pass
        Args:
            - x (Tensor): batched or unbatched image input with channel dimension
        returns: 
            - logits
        """
        for i in range(self.n_conv_layers):
            x = self.get_submodule(f'conv{i}')(x)
        return self.lin_layer(x)

    def regularize(self, p=1):
        """
        Computes p-norm over network parameters normalized to the number of parameters
        Args:
            - p (int): Order of matrix norm
        Returns:
            - p-norm of network parameters
        """
        norms = []
        for param in self.parameters():
            norms.append((torch.abs(param)**p).sum())
        return sum(norms)**(1/p)/self.n_params







class LiftingConv2d(nn.Module):
    def __init__(
        self,
        group,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        padding: int = 0,
        bias: bool = True,
        device: Union[torch.device, str] = torch.device('cpu')
    ):
        super().__init__()
        
        if type(kernel_size) is int:
            kernel_size = (kernel_size, ) * 2

        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # initialize the parameters
        normalization = 1/torch.tensor(out_channels * in_channels, device=device).sqrt()
        self.weight = nn.Parameter(
            normalization * torch.randn(
                size=(out_channels, in_channels) + kernel_size, 
                device=device
            ), 
            requires_grad=True
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros((out_channels), device=device), 
                requires_grad=True
            )
        else:
            self.bias = None
        
        self.group = group
        self.device = device
        self.to(device)
  
    def build_filter(self) -> torch.Tensor:
        """
        funtion to build the filter and bias for conv2d
        using the parameters weight and bias
        """
        # empty tensor for filter
        shape = (
            self.out_channels, 
            self.group.size,
            self.in_channels
        ) + self.kernel_size
        _filter = torch.empty(
            shape,
            device=self.device
        )
        # fill with four group representations of filters
        for index in self.group.indices:
            _filter[:,index] = self.group.action_on_grid(self.weight, index)

        # bias is shared -> just repeqt four times
        if self.bias is not None:
            _bias = torch.empty(
                (self.out_channels, self.group.size), 
                device=self.device
            )
            for index in self.group.indices:
                _bias[:,index] = self.bias
        else:
            _bias = None

        return _filter, _bias

    def __repr__(self):
        return f'LiftingConv2d({self.in_channels}, {self.out_channels}, group={self.group.name}, kernel_size={self.kernel_size})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        # reshape filters in order for conv2d to be usable
        _filter = _filter.reshape(
            self.out_channels * self.group.size,
            self.in_channels, 
            *self.kernel_size
        )
        _bias = _bias.reshape(self.out_channels * self.group.size)

        out = torch.conv2d(
            x, 
            _filter,
            padding=self.padding,
            bias=_bias
        )
        
        # reshape to actual shape
        return out.reshape(
            -1, 
            self.out_channels,
            self.group.size,
            out.shape[-2],
            out.shape[-1]
        )

class GroupConv2d(nn.Module):
    def __init__(
        self, 
        group,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        padding: int = 0,
        bias: bool = True,
        device: Union[torch.device, str] = torch.device('cpu')
    ):
        super().__init__()
        
        if type(kernel_size) is int:
            kernel_size = (kernel_size, ) * 2

        self.kernel_size = kernel_size
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

        normalization = 1/torch.tensor(out_channels * in_channels, device=device).sqrt()
        self.weight = torch.nn.Parameter(
            normalization * torch.randn(
                size=(
                    out_channels, 
                    in_channels,
                    group.size
                ) + kernel_size, 
                device=device
            ), 
            requires_grad=True
        )
        
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros((out_channels), device=device), 
                requires_grad=True
            )
        else:
            self.bias = None
        
        self.group = group
        self.device = device
        self.to(device)
  
    def build_filter(self) -> torch.Tensor:
        """
        funtion to build the filter and bias for conv2d
        using the parameters weight and bias
        """
        # empty tensor
        _filter = torch.empty(
            size=(
                self.out_channels, 
                self.group.size,
                self.in_channels, 
                self.group.size
            ) + self.kernel_size, 
            device=self.device
        )
        # fill with weights
        for index in self.group.indices:
            _filter[:,index] = self.group.action_on_group(self.weight, index)
    
        if self.bias is not None:
            _bias = torch.empty(
                size=(
                    self.out_channels, 
                    self.group.size
                ), 
                device=self.device
            )
            for index in self.group.indices:
                _bias[:,index] = self.bias
        else:
            _bias = None

        return _filter, _bias

    def __repr__(self):
        return f'GroupConv2d({self.in_channels}, {self.out_channels}, group={self.group.name}, kernel_size={self.kernel_size})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _filter, _bias = self.build_filter()

        # to be able to use torch.conv2d, we need to reshape
        # the filter and bias to stack together all filters
        _filter = _filter.reshape(
            self.out_channels * self.group.size, 
            self.in_channels * self.group.size, 
            *self.kernel_size
        )
        _bias = _bias.reshape(self.out_channels * self.group.size)
        x = x.view(x.shape[0], self.in_channels*self.group.size, x.shape[-2], x.shape[-1])

        out = torch.conv2d(
            x, 
            _filter,
            padding=self.padding,
            bias=_bias
        )
        
        return out.reshape(
            -1, 
            self.out_channels,
            self.group.size,
            out.shape[-2],
            out.shape[-1]
        )

class GroupCNN(nn.Module):
    def __init__(
        self,
        group,
        channels: List[int],
        kernel_sizes: Union[List[int], int],
        paddings: Union[List[int], int],
        pooling_kernels: List[Tuple[int, int, int]],
        pooling_strides: List[Tuple[int, int, int]],
        pooling_paddings: List[Tuple[int, int, int]],
        output_dims: Tuple[int],
        dropout: float = 0.0,
        sigmoid_out: bool = False,
        device: Union[torch.device, str] = torch.device('cpu')
    ):
        """
        p4-group CNN model
        Args:
            - channels (list): list of number of channels through layers
            - kernel_sizes (list of int or int): list of kernel sizes
                through layers, may be int if same for all layers
            - paddings (list of int or int): list of paddings through
                layers, may be int if same for all layers
            - pooling_kernels (List of Tuple of int): List of tuples of
                sizes for pooling. Has to be three tuples bc we pool
                over two spatial and one group pose dimension
            - pooling_strides (List of Tuple of int): List of tuples of
                strides for pooling.
            - pooling_paddings (List of Tuple of int): List of tuples
                of paddings for pooling.
            - output_dims (tuple of int): in and output sizes of linear
                output layer
            - dropout (float): dropout probability
            - device (torch.device or string): device
        """
        super().__init__()
        if type(kernel_sizes) is int:
            kernel_sizes = [kernel_sizes] * (len(channels)-1)
        if type(paddings) is int:
            paddings = [paddings] * (len(channels)-1)

        # input/lifting layer
        # self.conv0 = torch.nn.Sequential(
        #     LiftingConv2d(group, channels[0], channels[1], kernel_sizes[0], paddings[0], True, device),
        #     nn.MaxPool3d(pooling_kernels[0], pooling_strides[0], pooling_padding[0]),
        #     nn.ReLU()
        # )
        # hidden group conv layers
        for j, (ci, co, k, p, pk, ps, pp) in enumerate(zip(
            channels[:-1],
            channels[1:],
            kernel_sizes,
            paddings,
            pooling_kernels,
            pooling_strides,
            pooling_paddings)
        ):
            self.add_module(
                f'conv{j}',
                torch.nn.Sequential(
                    GroupConv2d(group, ci, co, k, p, True, device) if j > 0 else LiftingConv2d(group, ci, co, k, p, True, device),
                    nn.MaxPool3d(pk, ps, pp),
                    nn.ReLU()
                )
            )
        # linear output layer
        self.lin_layer = torch.nn.Sequential(
            nn.Flatten(1),
            nn.Linear(*output_dims),
            nn.Sigmoid() if sigmoid_out else nn.Identity()
        )
        
        self.n_conv_layers = len(kernel_sizes)
        self.n_params = sum([p.numel() for p in self.parameters()])
        
        self.device=device
        self.to(device)
    
    def forward(self, x):
        """
        forward pass
        Args:
            - x (Tensor): batched or unbatched image input with channel dimension
        returns: 
            - logits
        """
        for i in range(self.n_conv_layers):
            x = self.get_submodule(f'conv{i}')(x)
        return self.lin_layer(x)

    def regularize(self, p=1):
        """
        Computes p-norm over network parameters normalized to the number of parameters
        Args:
            - p (int): Order of matrix norm
        Returns:
            - p-norm of network parameters
        """
        norms = []
        for param in self.parameters():
            norms.append((torch.abs(param)**p).sum())
        return sum(norms)**(1/p)/self.n_params