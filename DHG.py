import requests 
import json

def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()

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
            self.dihypergraph["hyperedges"].append({
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

class DHGDeconstructor:
    def __init__(self, json_filepath):
        self.json_filepath = json_filepath
        self.dihypergraph = self.load_dihypergraph()

    def load_dihypergraph(self):
        with open(self.json_filepath, 'r') as json_file:
            data = json.load(json_file)
            return data

    def get_nodes(self):
        return self.dihypergraph.get('nodes', [])

    def get_hyperedges(self):
        return self.dihypergraph.get('hyperedges', [])

    def print_dihypergraph(self):
        print("Nodes:")
        for node in self.get_nodes():
            print(node)
        print("\nHyperedges:")
        for edge in self.get_hyperedges():
            print(edge)
                                
if __name__ == "__main__":
    BASE_URL = "https://api.github.com/repos/kennethZhangML/StochFlow/"
    TOKEN = read_api_key("/Users/kennethzhang/Desktop/gh-apikey.txt")

    HEADERS = {
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    my_dhg = DHGConstructor(BASE_URL, HEADERS, "stochflow")
    my_dhg.run_loop()

    json_filepath = "dihypergraph_stochflow.json"  
    deconstructor = DHGDeconstructor(json_filepath)
    deconstructor.print_dihypergraph()
