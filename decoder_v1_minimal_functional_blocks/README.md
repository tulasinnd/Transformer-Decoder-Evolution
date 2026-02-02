## Transformer Decoder Evolution — Version 1

This is a minimal decoder-only model for next-token prediction.The purpose of this version is to:

* Understand the basic forward pass through a decoder
* Inspect how a single input tensor moves through all layers
* Compute loss and propagate gradients in a backward pass

## Model Structure

The model implements the core blocks of a Transformer decoder:

1. **Causal Attention** – each token attends to itself and previous tokens
2. **Layer Normalization** – stabilizes training by normalizing token features
3. **Feed-Forward Network** – applies non-linear transformations token-wise
4. **Second Layer Normalization** – stabilizes the output of FFN
5. **Logit Projection** – maps hidden states to vocabulary logits

## How It Works

* A single random tensor is passed as input to the model
* Forward pass produces logits and attention weights
* Cross-entropy loss is computed against a random target tensor
* Backward pass updates model parameters
* Prints are included to inspect tensor flow, attention behavior, loss, and gradient propagation

This version does not include embeddings, positional encodings, multi-head attention, padding masks, or datasets. It focuses only on the minimal functional blocks of a decoder. This version is not meant for producing meaningful predictions.
