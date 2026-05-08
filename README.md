# Building GPT-2 from Scratch (PyTorch)

A modular, from-scratch implementation of the GPT-2 (Generative Pre-trained Transformer 2) architecture using PyTorch. This project focuses on understanding the underlying mechanics of the transformer decoder, including multi-head self-attention, feed-forward networks, and weight initialization.

## 🚀 Project Overview
This repository contains a clean, documented implementation of the GPT-2 model as described in the original OpenAI paper. The goal is to provide a transparent look at how tokens are processed, embedded, and transformed into predictive logits.

### Key Features
*   **Full Transformer Decoder:** Implementation of the GPT-2 block architecture.
*   **Multi-Head Attention:** Scaled dot-product attention with causal masking.
*   **Token & Positional Embeddings:** Learnable embeddings for both identity and position.
*   **Modular Design:** Easy-to-read code separating the attention, MLP, and block layers.

## 🏗️ Architecture
The model follows the standard GPT-2 configuration:
*   **GPT2Block:** The core building block consisting of Layer Normalization, Causal Self-Attention, and a Feed-Forward MLP.
*   **Causal Masking:** Ensures the model only attends to previous tokens (auto-regressive property).
*   **Weight Tieing:** Implementing weight sharing between the token embedding and the final linear projection layer.



## 🛠️ Installation & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch
*   NumPy
