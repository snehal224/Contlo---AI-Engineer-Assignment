import torch
import torch.nn as nn

# Define the hyperparameters and constants
num_layers = 12 # number of transformer decoder layers
num_heads = 12 # number of multi-heads in attention layer
dim = 768 # dimension of representation in each layer
rate = 4 # increase rate of dimensionality in bottleneck
vocab_size = 50257 # size of vocabulary
seq_len = 1024 # maximum sequence length
eos_token = 50256 # end-of-sequence token

# Create the GPT-2 model class
class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        # Create the token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        # Create the transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, num_heads, dim * rate, dropout=0.1),
            num_layers,
            nn.LayerNorm(dim)
        )
        # Create the output layer
        self.output = nn.Linear(dim, vocab_size, bias=False)
        # Tie the weights of the token embeddings and the output layer
        self.output.weight = self.token_emb.weight
    
    def forward(self, x):
        # Get the batch size and sequence length
        batch_size, seq_len = x.size()
        # Get the token and positional embeddings
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device)).unsqueeze(0).repeat(batch_size, 1, 1)
        # Add the token and positional embeddings
        emb = token_emb + pos_emb
        # Pass the embeddings to the transformer decoder
        output = self.decoder(emb, None)
        # Pass the output to the output layer
        logits = self.output(output)
        return logits

# Create the text generation function
def generate_text(model, prompt, max_len=100, temperature=1.0, top_k=0):
    # Set the model to evaluation mode
    model.eval()
    # Convert the prompt to tensor
    x = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    # Get the device of the model
    device = next(model.parameters()).device
    # Move the input to the device
    x = x.to(device)
    # Generate text until the end-of-sequence token or the maximum length is reached
    with torch.no_grad():
        for _ in range(max_len):
            # Get the logits from the model
            logits = model(x)
            # Get the logits of the last token
            logits = logits[:, -1, :] / temperature
            # Apply top-k filtering if specified
            if top_k > 0:
                # Get the top-k logits and indices
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                # Create a mask for the non-top-k indices
                mask = logits < top_k_logits[:, -1].unsqueeze(-1)
                # Set the non-top-k logits to a large negative value
                logits[mask] = -float('inf')
            # Sample a token from the logits
            token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            # Append the token to the input
            x = torch.cat([x, token], dim=-1)
            # Break the loop if the end-of-sequence token is generated
            if token.item() == eos_token:
                break
    # Return the generated text
    return x[0].tolist()
