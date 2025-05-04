import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Load and preprocess dataset
class NERDataset(Dataset):
    def __init__(self, file_path, word2idx, tag2idx, max_len=100):
        self.sentences, self.labels = self.load_data(file_path)
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len
    
    def load_data(self, file_path):
        sentences, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence, label = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence, label = [], []
                else:
                    parts = line.split()
                    sentence.append(parts[0])
                    label.append(parts[-1])
        return sentences, labels
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        word_indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in sentence]
        label_indices = [self.tag2idx[t] for t in label]
        
        pad_len = self.max_len - len(word_indices)
        if pad_len > 0:
            word_indices.extend([self.word2idx['<PAD>']] * pad_len)
            label_indices.extend([self.tag2idx['O']] * pad_len)
        else:
            word_indices = word_indices[:self.max_len]
            label_indices = label_indices[:self.max_len]
        
        return torch.tensor(word_indices), torch.tensor(label_indices)

# BiLSTM Model
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, dropout_prob1=0.2, dropout_prob2=0.3):
        super(BiLSTM_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout_prob1)  
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_prob2)  
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(self.dropout2(lstm_out))
        return output

# Training Setup
train_path = "en_ewt-ud-train.iob2"
dev_path = "en_ewt-ud-dev.iob2"

def build_vocab(file_path):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    tag2idx = {'O': 0}  # Ensure 'O' tag is included from the start

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if parts[0] not in word2idx:
                    word2idx[parts[0]] = len(word2idx)
                if parts[-1] not in tag2idx:
                    tag2idx[parts[-1]] = len(tag2idx)

    return word2idx, tag2idx


word2idx, tag2idx = build_vocab(train_path)

train_dataset = NERDataset(train_path, word2idx, tag2idx)
dev_dataset = NERDataset(dev_path, word2idx, tag2idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_NER(len(word2idx), 100, 50, len(tag2idx)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0  
    for words, labels in train_loader:
        words, labels = words.to(device), labels.to(device)
        optimizer.zero_grad()  
        outputs = model(words).permute(0, 2, 1) 
        loss = loss_function(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for words, labels in dev_loader:
        words, labels = words.to(device), labels.to(device)
        outputs = model(words)
        predictions = torch.argmax(outputs, dim=2)
        correct += (predictions == labels).sum().item()
        total += labels.numel()

accuracy = correct / total
print(f"Accuracy on dev data: {accuracy * 100:.2f}%")
