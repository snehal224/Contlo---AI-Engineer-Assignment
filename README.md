# Contlo---AI-Engineer-Assignment
# Task0:
# Task1:
To implement the GPT-2 model,
Import the necessary modules and libraries, such as torch, numpy, and tqdm.
Define the hyperparameters and constants, such as the number of layers, heads,
dimensions, vocabulary size, sequence length, learning rate, etc.
Create the token and positional embeddings, which are learned matrices that map each
token and position to a high-dimensional vector. You can use torch.nn.Embedding for
this purpose.
Implement the multi-head self-attention mechanism, which computes the similarity
between each pair of tokens and produces a weighted sum of their values. You can use
torch.nn.MultiheadAttention for this purpose.
Implement the point-wise feed-forward network, which consists of two linear layers with
a ReLU activation in between. You can use torch.nn.Linear and torch.nn.ReLU for this
purpose.
Implement the residual connection and layer normalization, which are techniques to
improve the stability and performance of the model. You can use torch.nn.ModuleList,
torch.nn.Sequential, and torch.nn.LayerNorm for this purpose.
Implement the transformer decoder layer, which combines the multi-head self-attention,
the point-wise feed-forward network, the residual connection, and the layer
normalization. You can use torch.nn.TransformerDecoderLayer for this purpose.
Implement the transformer decoder, which stacks multiple transformer decoder layers
and produces the output of the model. You can use torch.nn.TransformerDecoder for
this purpose.
Implement the output layer, which maps the output of the transformer decoder to the
vocabulary size and computes the logits. You can use torch.nn.Linear for this purpose.
Implement the GPT-2 model class, which integrates the token and positional
embeddings, the transformer decoder, and the output layer. You can use
torch.nn.Module for this purpose.
Implement the training loop, which iterates over the training data, feeds it to the model,
computes the loss, and updates the parameters. You can use torch.optim.Adam for the
optimizer and torch.nn.CrossEntropyLoss for the loss function.
Implement the evaluation loop, which iterates over the evaluation data, feeds it to the
model, computes the loss and other metrics, and saves the best model. You can use
torch.save and torch.load for saving and loading the model.
Implement the text generation function, which takes a prompt as input, feeds it to the
model, samples a token from the output distribution, and repeats until the end-ofsequence token is generated.

# Task2:
Rotary Positional Embedding:
In the original GPT-2 model, positional embeddings are added to the input embeddings to
provide the model with information about the position of tokens in a sequence. The Rotary
Positional Embedding is an alternative approach proposed by Su et al. in the RoFormer
paper.
Rotary Embeddings: Instead of using traditional sinusoidal positional embeddings,
Rotary Positional Embeddings utilize sine and cosine functions with a rotary parameter
(α). This introduces a more expressive way to encode positional information.
1.
Implementation: The implementation involves creating a RotaryPositionalEmbedding
module with learnable parameters α, and applying it to the input sequence. The rotation
is applied separately to even and odd indices of the positional embeddings.
2.
Potential Impact: Rotary embeddings aim to capture more intricate positional
relationships and dependencies, potentially improving the model's ability to understand
token positions.
3.
Group Query Attention:
In the original GPT-2 model, self-attention is a key mechanism where each token attends to
all other tokens in the sequence. Group Query Attention, as proposed in the GQA paper,
introduces a modification to the self-attention mechanism.
Implementation: The implementation involves modifying the MultiHeadAttention
module to group queries before computing attention scores. The grouped queries are
then concatenated before the final linear transformation.
1.
Potential Impact: Group Query Attention is designed to enhance the attention
mechanism by introducing structured patterns. It can potentially capture more
contextual information within each attention group.
2.
Additional Considerations:
Integration: These features need to be integrated into the GPT-2 model architecture
appropriately. This involves modifying the original GPT-2 model to incorporate the new
positional embeddings and attention mechanism.
Training: After integrating these features, the modified GPT-2 model needs to be
trained on a relevant dataset. Fine-tuning might be required to adapt the model to the
specific characteristics of the data.
Evaluation: The impact of these changes can be assessed through evaluation metrics,
comparing the performance of the modified model against the original GPT-2.
Hyperparameter Tuning: The effectiveness of these features can be sensitive to
hyperparameters. Experimentation with different values is often necessary.

# Task3:
The MNIST dataset is used for demonstration purposes only and is not suitable for the
GPT-2 model, which is designed for natural language processing tasks. For more
realistic examples, please refer to the official PyTorch tutorials and documentation.
The code assumes that the GPT-2 model class is defined in the same script as the training
function. Alternatively, the model class can be imported from another module or file.
The code uses the “gloo” backend for DDP and FSDP, which is suitable for CPU and
GPU communication. Other backends such as “nccl” or “mpi” can also be used
depending on the hardware and environment.
The code uses the default configuration for FSDP, which enables mixed precision and
CPU offloading. These options can be changed or customized according to the needs
and preferences of the user. For more details on FSDP configuration, please see the
PyTorch FSDP documentation.
The code does not include any logging, checkpointing, or validation logic, which are
essential for real-world training scenarios. These features can be easily added by using
the PyTorch utilities or custom functions.
The code does not handle any exceptions or errors that may occur during the training
process. The user should implement proper error handling and recovery mechanisms to
ensure the robustness and reliability of the training loop.
Benefits and drawbacks of each option:
Single GPU: This is the simplest and most straightforward option for training a model
on a single device. It does not require any additional setup or communication overhead.
However, it also has the most limitations in terms of scalability, performance, and
memory efficiency. It can only handle small to medium-sized models and datasets that
can fit on a single GPU.
DDP: It can handle large models and datasets that cannot fit on a single GPU by
splitting them across different devices and synchronizing the gradients and parameters. It
also leverages the parallelism and speedup that multiple devices can offer. However, it
also introduces some complexity and communication overhead. It requires setting up the
process group and the distributed sampler, and managing the device placement and data
loading. It also replicates the model and the optimizer states across all devices, which can
consume a lot of memory and bandwidth.
FSDP: It can handle very large models and datasets that cannot fit on a single GPU or
even multiple GPUs by sharding the model and the optimizer states across different
devices and only keeping a subset of them on each device. It also reduces the
communication overhead by using efficient collective operations and overlapping
communication and computation. However, it also introduces some challenges and
trade-offs. It requires more careful configuration and tuning, and may not be compatible
with some existing features or libraries.
