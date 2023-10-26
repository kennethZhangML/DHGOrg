import requests
import json 

TOKEN = "your-api-key"
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

BASE_URL = "https://api.github.com/repos/plurigrid/ontology/"

dihypergraph = {
    "nodes": [],
    "hyperedges": []
}

issues = requests.get(BASE_URL + "issues", headers = HEADERS).json()
for issue in issues:
    dihypergraph["nodes"].append({
        "id": f"issue_{issue['id']}",
        "type": "issue",
        "data": issue
    })
    dihypergraph["hyperedges"].append({
        "source": [f"user_{issue['user']['id']}"],
        "target": [f"issue_{issue['id']}"],
        "type": "created_by",
        "data": {}
    })

commits = requests.get(BASE_URL + "commits", headers = HEADERS).json()
for commit in commits:
    dihypergraph["nodes"].append({
        "id": f"commit_{commit['sha']}",
        "type": "commit",
        "data": commit
    })
    if commit['committer']:
        dihypergraph["hyperedges"].append({
            "source": [f"user_{commit['committer']['id']}"],
            "target": [f"commit_{commit['sha']}"],
            "type": "committed_by",
            "data": {}
        })
    else:
        pass

prs = requests.get(BASE_URL + "pulls", headers = HEADERS).json()
for pr in prs:
    dihypergraph["nodes"].append({
        "id": f"pushRequest_{pr['head']['sha']}",
        "type" : "pushRequest",
        "data": pr
    })

# Similarly, fetch PRs and emoji reactions and add them to nodes and hyperedges
# For simplicity, this script currently handles issues and commits.
# The approach can be extended to handle PRs, emoji reactions, and other entities.

with open("dihypergraph.json", "w") as outfile:
    json.dump(dihypergraph, outfile, indent = 4)

print("DiHypergraph saved to dihypergraph.json!")
