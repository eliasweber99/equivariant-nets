import torch
from torch_geometric import datasets

def label_to_onehot(labels, num_labels):
    I = torch.eye(num_labels)
    return I[labels]

def extract_subgraph(edge_index: torch.Tensor, rating: torch.Tensor, target_user: torch.Tensor, target_movie: torch.Tensor, h_hops: int = 1):
    """
    args:
        - edge_index (Tensor): Tensor of edge indices in the full graph
        - rating (Tensor): Tensor of ratings corresponding to the edges
        - target_user (Tensor): Tensor of shape (1, ) with the target users index
        - target_movie (Tensor): Tensor of shape (1, ) with the target movies index
        - h_hops (int): number of edges travelled when constructing the subgraph
    """
    # number of node labels for one hot encoding
    NUM_LABELS = 2*(h_hops+1)
    
    # construct adjacency matrix to get the edges later
    A = torch.zeros((edge_index[0].max()+1, edge_index[1].max()+1), dtype=torch.long)
    A[edge_index[0], edge_index[1]] = rating
    
    user_labels = torch.tensor([0], dtype=torch.long)
    movie_labels = torch.tensor([1], dtype=torch.long)
    
    users = target_user
    movies = target_movie
    
    previous_users = target_user
    previous_movies = target_movie
    
    # extract all nodes of the subgraph by hopping along the edges
    for h in range(h_hops):
        # create empty list for collecting
        new_users = torch.empty(size=(0,), dtype=torch.long)
        new_movies = torch.empty(size=(0,), dtype=torch.long)
        # loop through previous users to find new movies and vice versa
        for u in previous_users:
            # extract all movies that share an edge
            mask = edge_index[0] == u
            candidates = edge_index[1, mask]
            # check if they were already found before
            mask1 = (movies[:,None] != candidates[None]).all(dim=0)
            mask2 = (new_movies[:,None] != candidates[None]).all(dim=0)
            mask = torch.logical_and(mask1, mask2)
            # add only actually new nodes
            new_movies = torch.cat((new_movies, candidates[mask]), dim=0)
            # create a label depending on the hop distance
            movie_labels = torch.cat((movie_labels, (2*h+3)*torch.ones((len(new_movies), ), dtype=torch.long)), dim=0)
        for m in previous_movies:
            # extract all movies that share an edge
            mask = edge_index[1] == m
            candidates = edge_index[0, mask]
            # check if they were already found before
            mask1 = (users[:,None] != candidates[None]).all(dim=0)
            mask2 = (new_users[:,None] != candidates[None]).all(dim=0)
            mask = torch.logical_and(mask1, mask2)
            # add only actually new nodes
            new_users = torch.cat((new_users, candidates[mask]), dim=0)
            # create a label depending on the hop distance
            user_labels = torch.cat((user_labels, (2*h+2)*torch.ones((len(new_users),), dtype=torch.long)), dim=0)
        # add the new found users and movies to the full list
        users = torch.cat((users, new_users), dim=0)
        movies = torch.cat((movies, new_movies), dim=0)
        # set new as previous for next iteration
        previous_users = new_users
        previous_movies = new_movies
    # build reduced adj matrix (and remove edge betw target nodes)
    A[target_user, target_movie] = 0
    A_red = A[users][:,movies]
    # get edge indices of reduced system
    idxu, idxm = torch.where(A_red != 0)
    edge_index = torch.stack((idxu, idxm), dim=0)
    rating = A_red[idxu, idxm]
    # and repeat bc its an undirected graph (eg symmetric adjacency matrix)
    rating = torch.cat((rating, rating), dim=0)
    # edge types must be in {0, ..., num_types}
    edge_type = rating - 1
    # create feature matrices and append the nodes
    x_user = label_to_onehot(user_labels, NUM_LABELS)
    x_movie = label_to_onehot(movie_labels, NUM_LABELS)
    x = torch.cat((x_user, x_movie), dim=0)
    # adjust the edges st the indices for movies start with n_users
    n_users = len(user_labels)
    edge_index[1] += n_users
    # also here add the edges goind the same way
    edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)
    return (x, edge_index, edge_type), (torch.tensor([0], dtype=torch.long), torch.tensor([n_users], dtype=torch.long))

class ML100KDataset(torch.utils.data.Dataset):
    def __init__(self, root='./data', h_hops=1):
        super().__init__()
        pyg_dataset = next(iter(datasets.MovieLens100K(root)))
        self.edge_index = pyg_dataset['user', 'rates', 'movie'].edge_index
        self.rating = pyg_dataset['user', 'rates', 'movie'].rating
        self.h_hops = h_hops

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, index):
        target_nodes = self.edge_index[[0],index], self.edge_index[[1],index]
        data, target_nodes = extract_subgraph(self.edge_index, self.rating, *target_nodes, self.h_hops)
        return data, target_nodes, self.rating[index]