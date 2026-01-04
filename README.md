# Transformer From Scratch (PyTorch)

This repository contains a PyTorch implementation of a **Transformer model built from scratch**, focusing on the core architecture introduced in “Attention Is All You Need.” The notebook builds each major component manually instead of using high-level Transformer libraries.

## Script contains
- Complete Transformer implementation in a single Jupyter Notebook
- Token embeddings and positional encoding
- Scaled dot-product attention
- Multi-head self-attention mechanism
- Feed-forward neural network (FFN)
- Residual connections with Layer Normalization
- Encoder and Decoder layer stacks
- Padding mask and look-ahead (causal) mask
- Training and inference logic as demonstrated in the notebook

## What’s happening in the project
- Input tokens are converted into embeddings and combined with positional encodings to preserve sequence order.
- Self-attention computes relationships between all tokens using Query, Key, and Value projections.
- Multi-head attention captures different contextual relationships in parallel.
- The Encoder generates contextual representations of the input sequence.
- The Decoder uses masked self-attention and encoder–decoder attention to predict output tokens sequentially.
- Residual connections and layer normalization improve training stability.

## Key modules implemented
- Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Attention
- Feed Forward Network
- Encoder Layer and Encoder Stack
- Decoder Layer and Decoder Stack
- Full Transformer Architecture

## Requirements
Python 3.9+

### Main libraries used:
- torch
- numpy
- matplotlib
