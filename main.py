import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchtext')
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import random
import math
import time 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext 
from indicnlp.tokenize import indic_tokenize
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Load spaCy tokenizer
en_tokenizer = spacy.load('en_core_web_sm')

# Define tokenizers
def hindi_tokenizer(sentence):
    return [word.text for word in en_tokenizer.tokenizer(sentence.strip().split("-")[-1].strip())]

def english_tokenizer(sentence):
    return [word.text for word in en_tokenizer.tokenizer(sentence.strip().split("-")[-1].strip())]

# Load data
df = pd.read_csv('/Users/pray/Documents/Code-Mix/Dataset/train_final.txt', delimiter="\t", header=None)
df.rename(columns={0: 'english', 1: 'codemix'}, inplace=True)
train_data_sent, test_data_sent = train_test_split(df, test_size=0.2)
validation_data_sent = pd.read_csv('/Users/pray/Documents/Code-Mix/Dataset/dev_final.txt', delimiter="\t", header=None)
validation_data_sent.rename(columns={0: 'english', 1: 'codemix'}, inplace=True)

# Save data to JSON
train_data_sent.to_json('train_data.json', orient='records', lines=True)
validation_data_sent.to_json('validation_data.json', orient='records', lines=True)
test_data_sent.to_json('test_data.json', orient='records', lines=True)

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, field_name):
        self.data = pd.read_json(file_path, lines=True)
        self.tokenizer = tokenizer
        self.field_name = field_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx][self.field_name]
        return self.tokenizer(sentence)

# Instantiate datasets
train_dataset = CustomDataset('train_data.json', hindi_tokenizer, 'codemix')
validation_dataset = CustomDataset('validation_data.json', hindi_tokenizer, 'codemix')
test_dataset = CustomDataset('test_data.json', hindi_tokenizer, 'codemix')

print(len(train_dataset), len(validation_dataset), len(test_dataset))
# Example DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Function to build vocabulary from dataset
def build_vocab_from_dataset(dataset, tokenizer):
    def yield_tokens():
        for i in range(len(dataset)):
            tokens = dataset[i]  # Tokenized sentence as list of strings
            yield tokens
    
    return build_vocab_from_iterator(yield_tokens(), min_freq=10)

# Build vocabularies
hindi_vocab = build_vocab_from_dataset(train_dataset, hindi_tokenizer)
english_vocab = build_vocab_from_dataset(train_dataset, english_tokenizer)

print(f"Hindi Vocabulary Size: {len(hindi_vocab)}")
print(f"English Vocabulary Size: {len(english_vocab)}")

# Optional: Save vocabularies for future use
torch.save(hindi_vocab, 'hindi_vocab.pth')
torch.save(english_vocab, 'english_vocab.pth')

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.word_embeddings = nn.Embedding(input_size, embedding_size) 
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, input):
        embeddings = self.dropout(self.word_embeddings(input))
        o, h = self.gru(embeddings) 
        h = torch.tanh(self.linear(torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1)))
        return o, h
    
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, p, attention):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.word_embeddings = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size + 2 * hidden_size, hidden_size)
        self.linear = nn.Linear(embedding_size + 3 * hidden_size, output_size)
        self.output_size = output_size
        self.attention = attention
        self.hidden_size = hidden_size

    def forward(self, input, h, eo):        
        embeddings = self.dropout(self.word_embeddings(input.unsqueeze(0)))
        alpha = self.attention(h, eo).unsqueeze(1)
        eo = eo.permute(1, 0, 2)
        w = torch.bmm(alpha, eo).permute(1, 0, 2)
        o, h = self.gru(torch.cat((embeddings, w), dim = 2), h.unsqueeze(0))
        predictions = self.linear(torch.cat((o, w, embeddings), dim = 2).squeeze(0))
        return predictions, h.squeeze(0)

class Model(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()    
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
         
    def forward(self, input, actual):  
        eo, h = self.encoder(input)
        input = actual[0, :]
        predictions = torch.zeros(actual.shape[0], actual.shape[1], self.decoder.output_size).to(self.device)
        for t in range(1, actual.shape[0]):
            o, h = self.decoder(input, h, eo)
            predictions[t] = o
            predicted = o.argmax(1) 
            input = predicted
        return predictions      

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        scores = self.Va(torch.tanh(self.Wa(hidden) + self.Ua(encoder_outputs))) 
        scores = scores.squeeze(-1)
        weights = torch.nn.functional.softmax(scores, dim=1) 
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))  
        context = context.squeeze(1) 
        return context

def train(model, train_data_iterator, optimizer, criterion):   
    model.train()    
    total_loss = 0
    for batch in train_data_iterator:
        input = batch.english
        actual = batch.hindi
        optimizer.zero_grad()
        predictions = model(input, actual)
        vocab_size = predictions.shape[-1]
        predictions = predictions[1:].view(-1, vocab_size)
        actual = actual[1:].view(-1) 
        loss = criterion(predictions, actual)
        loss.backward()   
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()   
    average_loss = total_loss / len(train_data_iterator)
    return average_loss

def evaluate(model, data_iterator, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_iterator:
            input = batch.english
            actual = batch.hindi
            predictions = model(input, actual)
            vocab_size = predictions.shape[-1]
            predictions = predictions[1:].view(-1, vocab_size)
            actual = actual[1:].view(-1)
            loss = criterion(predictions, actual)
            total_loss += loss.item()
    average_loss = total_loss / len(data_iterator)
    return average_loss

enc = Encoder(len(english_vocab), 350, 512, 0.5)
attention = Attention(512)
dec = Decoder(len(hindi_vocab), 350, 512, 0.5,attention)

model = Model(enc, dec, device).to(device)
for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
model

# print(model)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = HINDI.vocab.stoi[HINDI.pad_token])

best_loss = 1e9

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Define collate function
def collate_fn(batch):
    sentences = [torch.tensor(item) for item in batch]  # Convert items to tensors
    sentences_padded = pad_sequence(sentences, padding_value=hindi_vocab['<pad>'], batch_first=True)
    return sentences_padded

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, pin_memory=True, drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, pin_memory=True, drop_last=False)

training_losses = []
validation_losses = []

for epoch in range(25):
    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss = evaluate(model, validation_loader, criterion)
    training_losses.append(np.exp(train_loss))
    validation_losses.append(np.exp(valid_loss))
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), 'encoder_decoder.pt')
    
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f"Train Loss (exponent to analyse better): {np.exp(train_loss):.3f}")
    print(f"Val. Loss (exponent to analyse better): {np.exp(valid_loss):.3f}")