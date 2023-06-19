import torch
from torch import nn
from torch.nn.init import uniform_
from torch_geometric.nn import Sequential
from torch_geometric.utils import dropout_edge

class RGCN(nn.Module):
    def __init__(
        self, in_features:int, out_features:int, num_relations:int, 
        aggr:str='mean', root:bool=True, bias:bool=True, dropout:float=0.0
    ):
        """
        args:
            :in_features: (int) number of incoming features
            :out_features: (int) number of outgoing features
            :num_relations: (int) number of different edge labels
            :aggr: (str) type of aggregation
            :root_weight: (bool) wether to add a root term
            :bias: (bool) wether to add a bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        if not aggr in ['mean', 'sum']:
            raise ValueError(":aggr: must be eiter 'mean' or 'sum'.")
        self.aggr = aggr
        self.dropout = dropout

        # initialize parameters
        bound = 1/torch.tensor(out_features).sqrt()

        weight = torch.empty(num_relations, out_features, in_features)
        uniform_(weight, -bound, bound)
        self.weight = nn.Parameter(weight)

        self.bias = None
        if bias:
            bias = torch.empty(out_features)
            uniform_(bias, -bound, bound)
            self.bias = nn.Parameter(bias)

        self.root = None
        if root:
            root = torch.empty(out_features, in_features)
            uniform_(root, -bound, bound)
            self.root = nn.Parameter(root)

    def forward(self, x, edge_index, edge_type):
        """
        args:
            :x: (num_nodes x features) Tensor of node features.
            :edge_index: (2 x num_edges) Tensor of edge indices.
                Must lead to a symmetric adjacency matrix.
            :edge_type: (num_edges) Tensor of edge labels. Values
                must be in {0, ..., num_relations}
        returns:
            Node features after propagating through layer.
        """
        # dropout
        if self.dropout:
            edge_index, dropout_mask = dropout_edge(
                edge_index, 
                self.dropout, 
                training=self.training
            )
            edge_type = edge_type[dropout_mask]
        # build adjacency matrices for every edge type
        n_nodes = len(x)
        adj = torch.zeros(self.num_relations, n_nodes, n_nodes)
        for l in edge_type.unique():
            mask = edge_type == l
            adj[l, edge_index[0, mask], edge_index[1, mask]] = 1
        # if mean aggregation put 1/N instead of ones in the adjacency matrix
        if self.aggr == 'mean':
            norm = 1/adj.sum(dim=-1, keepdim=True)
            norm = norm.nan_to_num(.0, .0, .0)
            adj = norm*adj
        # output
        out = torch.einsum('bik,bkl,bml->bim', adj, x[None], self.weight).sum(0)
        if not self.root is None:
            out += x@self.root.T
        if not self.bias is None:
            out += self.bias[None]
        return out


class GraphNet(nn.Module):
    def __init__(
        self, 
        latent_dims, 
        h_hops, 
        num_relations, 
        dropout_conv=.2, 
        dropout_lin=.5, 
        device=torch.device('cpu')
    ):
        """
        stuff
        """
        super().__init__()
        self.conv0 = Sequential(
            'x, edge_index, edge_type',
            [(RGCN(2*(h_hops + 1), latent_dims[0], num_relations, dropout=dropout_conv), 'x, edge_index, edge_type -> x'),
            nn.ReLU()]
        )
        for k, (i, o) in enumerate(zip(latent_dims[:-1], latent_dims[1:])):
            self.add_module(
                f'conv{k+1}',
                Sequential(
                    'x, edge_index, edge_type',
                    [(RGCN(i, o, num_relations, dropout=dropout_conv), 'x, edge_index, edge_type -> x'),
                    nn.ReLU()]
                )
            )
        self.lin = nn.Sequential(
            nn.Linear(2*sum(latent_dims), 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_lin),
            nn.Linear(128, 1)
        )
        self.h_hops = h_hops
        self.network_depth = len(latent_dims)
        # self.device = device
        self.to(device)
    
    def forward(self, x, edge_index, edge_type, tgt_u, tgt_m):
        cats = []
        for i in range(self.network_depth):
            x = self.get_submodule(f'conv{i}')(x, edge_index, edge_type)
            cats.append(x)
        x = torch.cat(cats, dim=1)
        x = torch.cat((x[tgt_u], x[tgt_m]), dim=1).squeeze()
        return self.lin(x)