import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

# 将邻接矩阵转为pyG要求格式
def convert_adj(adj):
    data_adj = [[],[]]
    data_edge_features = []
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i,j] != 0:
                data_edge_features.append(adj[i,j])
                data_adj[0].append(i)
                data_adj[1].append(j)
    data_adj = torch.tensor(data_adj, dtype=torch.int64)
    data_edge_features = torch.Tensor(data_edge_features)
    return data_adj, data_edge_features


class GAT(torch.nn.Module):
    def __init__(self, device, in_features, out_features, edge_dim=1):
        super(GAT, self).__init__()
        self.gat = GATConv(in_features, out_features, edge_dim=edge_dim).to(device)
        self.convert = convert_adj
        self.device = device

    def forward(self, data, adj):
        adj = adj.cpu()
        data = data.cpu()
        data_adj, data_edge_features = self.convert(adj)
        data_list = [Data(x=x_, edge_index=data_adj, edge_attr=data_edge_features) for x_ in data]
        batch = Batch.from_data_list(data_list)
        batch.to(self.device)
        x, attention_weights = self.gat(batch.x, batch.edge_index, batch.edge_attr, return_attention_weights=True)
        x = x.view(len(data), len(adj), -1)
        return x, attention_weights

