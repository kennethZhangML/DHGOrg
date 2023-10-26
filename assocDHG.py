import networkx as nx
import matplotlib.pyplot as plt
import json

class DHGVisualizer:
    def __init__(self, dhg_filename):
        with open(dhg_filename, 'r') as f:
            self.dihypergraph = json.load(f)
        self.graph = nx.DiGraph()

    def create_graph(self):
        for node in self.dihypergraph['nodes']:
            self.graph.add_node(node['id'], type=node['type'], data=node['data'])

        for edge in self.dihypergraph['hyperedges']:
            source = edge['source'][0]  
            for target in edge['target']:
                self.graph.add_edge(source, target, type=edge['type'])

    def visualize(self):
        no_type_nodes = [node for node in self.graph.nodes if 'type' not in self.graph.nodes[node]]
        if no_type_nodes:
            print(f"Nodes without 'type': {no_type_nodes}")
            return

        pos = nx.spring_layout(self.graph)
        color_map = {'issue': 'blue', 'commit': 'green', 'pushRequest': 'red', 'user': 'yellow'}
        colors = [color_map[self.graph.nodes[node]['type']] for node in self.graph.nodes]
        
        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, node_color=colors, with_labels=True)
        plt.show()

    def compute_associations(self):
        issue_nodes = [node['id'] for node in self.dihypergraph['nodes'] if node['type'] == 'issue']
        associations = {}

        for i in range(len(issue_nodes)):
            for j in range(i+1, len(issue_nodes)):
                try:
                    path_length = nx.shortest_path_length(self.graph, issue_nodes[i], issue_nodes[j])
                    associations[(issue_nodes[i], issue_nodes[j])] = path_length
                except nx.NetworkXNoPath:
                    pass
        return associations


if __name__ == "__main__":
    dhg_vis = DHGVisualizer("dihypergraph_my_sample.json")
    dhg_vis.create_graph()
    dhg_vis.visualize()

    associations = dhg_vis.compute_associations()
    print(associations)
