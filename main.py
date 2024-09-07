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
        return torch.tensor(self.tokenizer(sentence), dtype=torch.long)

# Instantiate datasets
train_dataset = CustomDataset('train_data.json', hindi_tokenizer, 'codemix')
validation_dataset = CustomDataset('validation_data.json', hindi_tokenizer, 'codemix')
test_dataset = CustomDataset('test_data.json', hindi_tokenizer, 'codemix')

# Example DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(len(train_dataset), len(validation_dataset), len(test_dataset))
