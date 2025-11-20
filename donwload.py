from torch_geometric.datasets import Flickr
import torch_geometric.transforms as T

dataset = Flickr(root="./data/Flickr", transform=T.NormalizeFeatures())
data = dataset[0]

print(data)
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")