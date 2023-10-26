import torch 
import torch.nn.functional as F 
import torch
from torch_geometric.data import Data

from DHG import *

def load_dihypergraph_from_json(json_filepath):
    with open(json_filepath, 'r') as json_file:
        dihypergraph = json.load(json_file)
    return dihypergraph

dihypergraph = load_dihypergraph_from_json("dihypergraph_my_sample.json")

import torch
from torch_geometric.data import Data

node_mapping = {node['id']: idx for idx, node in enumerate(dihypergraph['nodes'])}
issue_indices = [i for i, node in enumerate(dihypergraph['nodes']) if node['type'] == 'issue']

edge_index = []
for edge in dihypergraph['hyperedges']:
    source = node_mapping[edge['source'][0]]  
    target = node_mapping[edge['target'][0]]
    edge_index.append([source, target])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
data = Data(x=torch.randn((len(issue_indices), 32)), edge_index = edge_index)

from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(32, 64, heads=2, dropout=0.6)
        self.conv2 = GATConv(64*2, 1, heads=1, concat=True, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GAT()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = torch.mean(out)
    loss.backward()
    optimizer.step()





