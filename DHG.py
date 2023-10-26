import requests 
import json

TOKEN = "your-api-key"
HEADERS = {
    "Authorization": f"Token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

dihypergraph = {
    "nodes": [],
    "hyperedges": []
}

class DHGConstructor:
    def __init__(self, BASE_URL, HEADERS, dhg_name, init_empty = True, dhg = None):
        self.dhg_name = dhg_name
        self.issues = requests.get(BASE_URL + "issues", headers = HEADERS).json()
        self.commits = requests.get(BASE_URL + "commits", headers = HEADERS).json()
        self.prs = requests.get(BASE_URL + "pulls", headers = HEADERS).json()

        self.BASE_URL = BASE_URL
        self.HEADERS = HEADERS

        if init_empty == True:
            self.dihypergraph = {
                "nodes": [],
                "hyperedges": []
            }
        else:
            self.dihypergraph = dhg

    def run_loop(self):
        for issue in self.issues:
            self.dihypergraph["nodes"].append({
                "id": f"issue{issue['id']}",
                "type": "issue",
                "data": issue
            })
            self.dihypergraph["hyperedges"].append({
                "source": [f"user_{issue['user']['id']}"],
                "target": [f"issue_{issue['id']}"],
                "type": "created_by",
                "data": {}
            })
        
        for commit in self.commits:
            self.dihypergraph["nodes"].append({
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

        for pr in self.prs:
            self.dihypergraph["nodes"].append({
                "id": f"pushRequest_{pr['head']['sha']}",
                "type" : "pushRequest",
                "data": pr
            })
        
        with open(f"dihypergraph_{self.dhg_name}.json", "w") as outjson:
            json.dump(self.dihypergraph, outjson, indent = 4)
        print(f"Dihypergraph constructed to dihypergraph_{self.dhg_name}")
                                
if __name__ == "__main__":
    BASE_URL = "https://api.github.com/repos/plurigrid/ontology/"
    TOKEN = "your-api-token"
    HEADERS = {
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    my_dhg = DHGConstructor(BASE_URL, HEADERS, "my_sample")
    my_dhg.run_loop()
