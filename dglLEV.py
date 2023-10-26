import dgl 
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

import json 

def convert_to_dgl_graph(dihypergraph):
    g = dgl.DGLGraph()
    g.add_nodes(len(dihypergraph["nodes"]))
    
    for edge in dihypergraph["hyperedges"]:
        for src in edge["source"]:
            for tgt in edge["target"]:
                g.add_edge(src, tgt)
    return g

class SimpleGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, h_feats)
        self.conv2 = dglnn.GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        h = self.conv2(g, h)
        return h

if __name__ == "__main__":
    with open("dihypergraph_my_sample.json", "r") as file:
        dihypergraph = json.load(file)