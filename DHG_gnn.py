import torch
import json
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)  

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim = 1)

model = GNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
labels = x.long().squeeze()  

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, labels)
    loss.backward()
    optimizer.step()

model.eval()
print(model(data))
