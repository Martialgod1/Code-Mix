import pandas as pd
import numpy as np
import spacy
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from indicnlp.tokenize import indic_tokenize
# from torchtext.legacy import data
# from torchtext.legac

df = pd.read_csv('/Users/pray/Documents/Code-Mix/Dataset/R11_final_data/train_final.txt', delimiter = "\t", header = None)
df.rename(columns = {0 : 'english', 1 : 'codemix'}, inplace = True)
en_tokenizer = spacy.load('en_core_web_sm')
# print(df)

train_data_sent, test_data_sent = train_test_split(df, test_size = 0.2)
validation_data_sent = pd.read_csv('/Users/pray/Documents/Code-Mix/Dataset/R11_final_data/dev_final.txt', delimiter = "\t", header = None)
validation_data_sent.rename(columns = {0 : 'english', 1 : 'codemix'}, inplace = True)
# train_data_sent
# validation_data_sent=validation_data_sent.iloc[:5]
validation_data_sent







