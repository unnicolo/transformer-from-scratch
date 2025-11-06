# Implementing a Transformer From Scratch with PyTorch  

A practical deep learning project focused on building a **Transformer** model from scratch using **PyTorch**, without relying on high-level libraries like `torch.nn.Transformer`. The project is designed to deepen theoretical understanding through hands-on implementation of attention mechanisms, positional encoding, and encoder-decoder architecture.

## 🔧 Technologies Used

- **Language:** Python 3.12.0 
- **Frameworks:** PyTorch, NumPy  
- **Tools:** VSCode, Git  
- **Key Concepts:**  
  - Attention Mechanism  
  - Multi-Head Attention  
  - Layer Normalization  
  - Positional Encoding  
  - Masking Techniques  
  - Encoder-Decoder Architecture  

### 📁 Project Structure

```plaintext
transformer-from-scratch/
|
├── data/
│   ├── __init__.py           
│   ├── batch.py               # Holds a batch training data
│   ├── synthetic_data.py      # Generation of synthetic training data
|
├── models/
│   ├── __init__.py           
│   ├── attention.py           # Scaled dot-product & multi-head attention
│   ├── encoder.py             # Transformer encoder block
│   ├── decoder.py             # Transformer decoder block
│   ├── generator.py           # Linear layer + softmax output probabilities generation step
│   ├── transformer.py         # Full model integration
│   ├── feed_forward.py        # Position-wise feed-forward network
│   ├── embeddings.py          # Token and positional embeddings
|   └── utils.py               # Helper functions, masking, cloning etc 
|
├── train/                     # (Planned) Model training
│   ├── __init__.py           
│   ├── loss.py                # (Planned) Loss computation
│   ├── rate.py                # Implementation of the NoamOptimizer, adjusting the learning rate.
│   ├── state.py               # TrainState dataclass
│   ├── train.py               # Training loop
|
├── main.py                    # Sample forward pass & debugging
├── inference.py               # (Planned) Inference script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Licensing

The source code in this project is released under the MIT License.  
You are free to use, modify, and distribute this software under the terms of the license.

See the full license text in the [LICENSE](LICENSE) file.

