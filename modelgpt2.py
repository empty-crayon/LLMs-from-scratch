import torch
import torch.nn as nn
import math

# Components: Causal Self-Attention, FeedForward, TransformerBlock as defined above
class CausalSelfAttention(nn.Module):
    
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key   = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, x):
        B, T, C = x.shape
        Q = self.query(x);  K = self.key(x);  V = self.value(x)
        # split into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # scaled dot-product attention
        att = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, V)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # reassemble all heads
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, emb_dim, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, expansion * emb_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expansion * emb_dim, emb_dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(emb_dim, num_heads)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffn = FeedForward(emb_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # Self-attention + residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        # Feed-forward + residual
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
    
    @torch.no_grad()
    def generate(model, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.max_seq_len:]         # respect context window
            logits = model(idx_cond)[:, -1, :]             # last position
            next_id = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, max_seq_len, emb_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.fc_out = nn.Linear(emb_dim, vocab_size)
        self.max_seq_len = max_seq_len
    def forward(self, idx):
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError("Sequence length exceeds context length")
        # Embedding and position encoding
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        # Final norm and output
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits

# Example usage with small dimensions for demonstration
vocab_size = 100
model = GPT2Model(vocab_size=vocab_size, max_seq_len=20, emb_dim=16, num_layers=2, num_heads=4)
dummy_input = torch.randint(0, vocab_size, (1, 5))  # a single sequence of length 5
logits = model(dummy_input)
print("Logits shape:", logits.shape)  # should be (1, 5, 100)
