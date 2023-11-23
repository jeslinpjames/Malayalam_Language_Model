import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 8
batch_size = 32
max_iterations = 5000
learning_rate = 1e-3
device = 'cuda'if torch.cuda.is_available()else 'cpu'


torch.manual_seed(1337)

with open('text.txt', 'r', encoding='utf-8' ) as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Vocab size:', vocab_size)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

data = torch.tensor(encode(text),dtype=torch.long, device=device)
n = int (0.9*len(data))
train_data = data[:n]
val_data = data[n:]
