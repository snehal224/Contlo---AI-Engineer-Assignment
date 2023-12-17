import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.cos_position = nn.Parameter(torch.zeros(max_len, d_model // 2), requires_grad=False)
        self.sin_position = nn.Parameter(torch.zeros(max_len, d_model // 2), requires_grad=False)

        self.init_weights()

    def init_weights(self):
        self.cos_position.data[:, 0::2] = torch.cos(self.alpha * torch.arange(0, self.d_model, 2).float() / self.d_model)
        self.sin_position.data[:, 1::2] = torch.sin(self.alpha * torch.arange(1, self.d_model, 2).float() / self.d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), dtype=torch.float, device=x.device).unsqueeze(0)
        angles = self.alpha * positions

        pos_embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_embedding = pos_embedding[:, :x.size(1)].unsqueeze(0)

        return x + pos_embedding

# Example usage:
rotary_pos_emb = RotaryPositionalEmbedding(d_model=768, max_len=512)
input_sequence = torch.randn(1, 10, 768)  # Replace with your input sequence
output_sequence = rotary_pos_emb(input_sequence)
print(output_sequence)

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(GroupQueryAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.group_size = n_heads // 2

    def forward(self, Q, K, V, mask=None):
        Q = self.W_Q(Q).view(-1, Q.size(1), self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(-1, K.size(1), self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(-1, V.size(1), self.n_heads, self.d_k).transpose(1, 2)

        Q = torch.cat(torch.split(Q, self.group_size, dim=2), dim=2)
        K = torch.cat(torch.split(K, self.group_size, dim=2), dim=2)
        V = torch.cat(torch.split(V, self.group_size, dim=2), dim=2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(-1, x.size(2), self.n_heads * self.d_k)
        x = self.W_O(x)
        return x

# Example usage:
group_query_attn = GroupQueryAttention(d_model=768, n_heads=12)
input_sequence = torch.randn(1, 10, 768)  # Replace with your input sequence
output_sequence = group_query_attn(input_sequence, input_sequence, input_sequence)
print(output_sequence)
