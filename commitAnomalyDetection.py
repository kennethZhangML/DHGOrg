from DHG import DHGConstructor
from transformers import DistilBertTokenizer
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

import json
import torch 
from torch.optim import Adam

LEARNING_RATE = 0.001
EPOCHS = 10

with open("dihypergraph_my_sample.json", "r") as file:
    dihypergraph = json.load(file)
commit_messages = [node["data"]["commit"]["message"] for node in dihypergraph["nodes"] if node["type"] == "commit"]
print(f"Number of Commits: {len(commit_messages)}")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 128
commit_inputs = tokenizer(commit_messages, padding = 'max_length', truncation = True, max_length = MAX_LEN, return_tensors = "pt")

class TransformerAutoencoder(nn.Module):
    def __init__(self):
        super(TransformerAutoencoder, self).__init__()
        config = DistilBertConfig(max_position_embeddings = 128)
        self.encoder = DistilBertModel(config)
        self.decoder = DistilBertModel(config)
    
    def forward(self, input_ids, attention_mask = None):
        hidden_states = self.encoder(input_ids, attention_mask = attention_mask).last_hidden_state
        outputs = self.decoder(inputs_embeds = hidden_states, attention_mask = attention_mask).last_hidden_state
        return outputs
    
model = TransformerAutoencoder()
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr = LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(commit_inputs["input_ids"], attention_mask = commit_inputs["attention_mask"])
    loss = criterion(outputs, model.encoder(commit_inputs["input_ids"]).last_hidden_state)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item()}")
model.eval()

reconstructed = model(commit_inputs["input_ids"], attention_mask = commit_inputs["attention_mask"])
losses = []
for original, recon in zip(commit_inputs["input_ids"], reconstructed):
    loss = criterion(recon, model.encoder(original).last_hidden_state).item()
    losses.append(loss)
threshold = sum(losses) / len(losses)
anomalies = [idx for idx, loss in enumerate(losses) if loss > threshold * 1.5] 

print(f"Found {len(anomalies)} anomalous commits!")



