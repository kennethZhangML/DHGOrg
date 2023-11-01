from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv  
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch_sparse

import torch 
import torch.nn.functional as F
import json 

with open("dihypergraph_conversation.json", "r") as file:
    dhg = json.load(file)

nodes = {node["id"]: i for i, node in enumerate(dhg["nodes"])}
edge_source, edge_target = [], []

for hyperedge in dhg["hyperedges"]:
    source = nodes[hyperedge["source"][0]]
    target = nodes[hyperedge["target"][0]]
    edge_source.append(source)
    edge_target.append(target)      

x = torch.tensor([[1] if node["type"] == "message" else [0] for node in dhg["nodes"]], dtype = torch.float)
edge_index = torch.tensor([edge_source, edge_target], dtype = torch.long)

data = Data(x = x, edge_index = edge_index)

data = T.TwoHop()(data)

class LinkPredictionModel(torch.nn.Module):
    def __init__(self):
        super(LinkPredictionModel, self).__init__()
        self.conv1 = SAGEConv(1, 16)
        self.conv2 = SAGEConv(16, 16)
    
    def forward(self, x, adjs):
        for adj in adjs:
            x = self.conv1(x, adj.t())
            x = F.relu(x)
            x = F.dropout(x, training = self.training)
        return x

def get_loss(pos_out, neg_out):
    pos_loss = F.logsigmoid(pos_out).mean()
    neg_loss = F.logsigmoid(-neg_out).mean()
    return -pos_loss - neg_loss

model = LinkPredictionModel()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
train_loader = NeighborSampler(edge_index, sizes = [10, 10], batch_size = 128, shuffle = True)

model.train()
for epoch in range(200):
    for batch_size, n_id, adjs in train_loader:
        optimizer.zero_grad()
        
        pos_out = model(data.x[n_id], adjs)
        pos_out = pos_out.squeeze()[adjs[0][0].argmin()]
        neg_out = pos_out.flip(0)
        
        loss = get_loss(pos_out, neg_out)
        loss.backward()
        optimizer.step()
