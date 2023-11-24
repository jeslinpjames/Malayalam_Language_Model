import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 8 #number of characters per sequence
batch_size = 32 #number of sequences per batch
max_iterations = 8000
learning_rate = 1e-4
device = 'cuda'if torch.cuda.is_available()else 'cpu'
print(device)
n_embed = 32
eval_interval = 300
eval_iters = 300


torch.manual_seed(1337)

#data extraction
with open('text.txt', 'r', encoding='utf-8' ) as f:
    text = f.read()

chars = sorted(list(set(text)))#makes a list of all unique characters in the text
vocab_size = len(chars)#vocubulary size of the entire text
print('Vocab size:', vocab_size)

stoi = {ch:i for i,ch in enumerate(chars)}#string to index
itos = {i:ch for i,ch in enumerate(chars)}#index to string
encode = lambda x: [stoi[ch] for ch in x]#To convert a string to corresponding interger tokens
decode = lambda x: ''.join([itos[i] for i in x])#Converts a list of interger tokens to string

data = torch.tensor(encode(text),dtype=torch.long, device=device)
n = int (0.9*len(data))#Takes 90% data to train
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split =='train'else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))#gives batch_size number of incides inside the data to split into blocks
    x = torch.stack([data[i:i+block_size] for i in ix])#creates batch_size number of blocks
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])#create blocks shifted by 1
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses =torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k]= loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size, bias = False)
        self.query = nn.Linear(n_embed,head_size,bias = False)
        self.value = nn.Linear(n_embed,head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))#since tril isn't a parameter of nn.Module, so we register it as a buffer

    def forward(self,x):
        B,T,C = x.shape #Batch, Time, Channels
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**(-0.5) #(B,T,C) @ (B,C,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) #B,T,T
        wei = F.softmax(wei,dim =-1)#B,T,T
        v = self.value(x) #(B,T,C)
        out = wei @ v #(B,T,T) @ (B,T,C) = (B,T,C)
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)#Embedding layer
        self.position_embedding_table = nn.Embedding(block_size,n_embed)#position embedding layer
        self.lm_head = nn.Linear(n_embed,vocab_size)#Linear layer
        self.sa_head = Head(n_embed)
    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) #(Batch,Time ,Channels)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = pos_emb + tok_emb
        x = self.sa_head(x)
        logits =self.lm_head(x)#(B,T,Vocab_size)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)# this is done so that logits can be passed to cross entropy loss
            targets = targets.view(B*T)# can also be(-1) pytorch will automatically infer the shape
            loss = F.cross_entropy(logits,targets)# cross entropy loss
        return logits, loss
    def generate(self,idx,max_new_tokens):
        for i in range(max_new_tokens):
            # idx_cont = idx[:,-block_size:]#crop idx to last block_size tokens
            logits,loss = self.forward(idx)
            logits = logits[:,-1,:] # to drop the last dimension T
            probs = F.softmax(logits, dim=-1)# converted to probabilities
            idx_next = torch.multinomial(probs,num_samples=1)# to give just 1 after sampling from the probs
            idx = torch.cat((idx,idx_next), dim =1)# append next integer to current stream of intergers after sampling 1 element
        return idx
    
model  = BigramLanguageModel()
m = model.to(device)

Optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iteration in range(max_iterations):
    if iteration % eval_interval==0:
        losses = estimate_loss()
        print(f"step {iteration} : train loss {losses['train']:.4f},val loss {losses['val']:.4f}")
    xb,yb = get_batch('train')
    logits,loss = m(xb,yb)
    loss.backward()#backpropagation
    Optimizer.step()#gradient descent
print('Loss:',loss.item())
content = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(idx=content,max_new_tokens =300)[0].tolist()))