import json

class ConversationDHGConstructor:
    def __init__(self, dhg_name):
        self.dhg_name = dhg_name
        self.dihypergraph = {
            "nodes": [],
            "hyperedges": []
        }
        self.node_counter = 0
    
    def add_message(self, sender, message):
        sender_id = f"user_{sender}"
        message_id = f"message_{self.node_counter}"

        if not any(node["id"] == sender_id for node in self.dihypergraph["nodes"]):
            self.dihypergraph["nodes"].append({
                "id": sender_id,
                "type": "user",
                "data": {"name": sender}
            })
        
        self.dihypergraph["nodes"].append({
            "id": message_id,
            "type": "message",
            "data": {"content": message}
        })

        self.dihypergraph["hyperedges"].append({
            "source": [sender_id],
            "target": [message_id],
            "type": "sent_by",
            "data": {}
        })

        self.node_counter += 1
    
    def save_to_file(self):
        with open(f"dihypergraph_{self.dhg_name}.json", "w") as outjson:
            json.dump(self.dihypergraph, outjson, indent = 4)
        print(f"Dihypergraph constructed to dihypergraph_{self.dhg_name}.json")

if __name__ == "__main__":
    conversation_dhg = ConversationDHGConstructor("conversation")
    
    conversation_dhg.add_message("User", "Hello, GPT!")
    conversation_dhg.add_message("GPT", "Hello! How can I assist you today?")
    conversation_dhg.add_message("User", "Can you help me with my code?")
    conversation_dhg.add_message("GPT", "Of course! What do you need help with?")
    
    conversation_dhg.save_to_file()
