import json
import networkx as nx
import pandas as pd 

import matplotlib.pyplot as plt

def visualize_graph(graph):
    plt.figure(figsize=(10, 10))  
    pos = nx.spring_layout(graph)  
    labels = {node: data['title'] for node, data in graph.nodes(data = True)}    
    nx.draw(graph, pos, labels = labels, with_labels = True, node_size = 2000, node_color = 'skyblue', font_size = 8, font_color = 'black')    
    plt.title("Issue Graph")
    plt.show()

def issue_data(node):
    title = node['data']['title']
    body = node['data']['body']
    id = node['data']['id']
    created_at = node['data']['created_at']
    closed_at = node['data']['closed_at']
    return title, body, id, created_at, closed_at

if __name__ == "__main__":
    org_dhg = nx.Graph()
    issue_dhg = nx.Graph()
    commit_dhg = nx.Graph()

    json_file = open("dihypergraph_plurigrid_dhg.json")
    json_str = json_file.read()
    json_data = json.loads(json_str)
    datapoints = json_data["nodes"]

    print(len(datapoints))

    for node in datapoints:
        if node['type'] == "issue":
            title, body, id, created_at, closed_at = issue_data(node)
            print("Title: ", title)
            print("Body: ", body)
            print("ID: ", id)
            print("Created At: ", created_at)
            print("Closed At: ", closed_at)

            issue_id = f"issue_{id}"

            issue_dhg.add_node(issue_id, title = title, body = body, 
                               created_at = created_at, closed_at = closed_at)
    
    visualize_graph(issue_dhg)


        

                
