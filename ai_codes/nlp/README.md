# NLP Foundations

A progressive learning journey through Natural Language Processing, tracing the evolution of techniques from sparse lexical representations to modern attention-based architectures. Each notebook is fully executable, well-documented, and implements concepts from first principles.

---

## Learning Journey

```
TF-IDF (Sparse Vectors)
       ↓
Word Vectors (Dense Representations)
       ↓
Word Embeddings (Learned Representations)
       ↓
Recurrent Neural Networks (Sequence Modeling)
       ↓
LSTM (Long-term Dependencies)
       ↓
Attention Mechanisms (Selective Focus)
       ↓
Self-Attention (Token Interactions)
       ↓
Transformers (Next Section)
```

---

## Folder Structure

```
nlp/
├── 01-tf-idf/
│   ├── README.md
│   └── notebook.ipynb
├── 02-kelime-vektorleri/
│   ├── README.md
│   └── notebook.ipynb
├── 03-word-embeddings/
│   ├── README.md
│   └── notebook.ipynb
├── 04-rnn/
│   ├── README.md
│   └─��� notebook.ipynb
├── 05-lstm/
│   ├── README.md
│   └── notebook.ipynb
├── 06-attention-mechanisms/
│   ├── README.md
│   └── notebook.ipynb
├── 07-self-attention/
│   ├── README.md
│   └── notebook.ipynb
└── README.md (this file)
```

---

## Learning Objectives

By working through these notebooks, you will understand:

- **Text Representation**: How to convert text into numerical formats (sparse and dense)
- **Semantic Similarity**: Measuring meaning through vector spaces and embeddings
- **Sequence Modeling**: Processing temporal dependencies in language
- **Attention Mechanisms**: Focusing on relevant information in complex inputs
- **Deep Learning for NLP**: Building and training neural networks for language tasks
- **Architecture Evolution**: How limitations of each technique motivated the next

---

## Notebook Overview

| Notebook | Main Concepts | Key Implementation | Visualizations | Learning Focus |
|----------|---------------|-------------------|-----------------|----------------|
| **01. TF-IDF** | Bag of Words, Term Frequency, Sparse Vectors | Document similarity search engine | Term heatmaps, frequency distributions | Foundation: discrete features |
| **02. Word Vectors** | Dense representations, semantic similarity, word analogies | Vector space arithmetic, PCA projection | Embedding space visualization | Transition: sparse → dense |
| **03. Word Embeddings** | Word2Vec, CBOW, Skip-Gram, Gensim | Training embeddings from scratch | Semantic clustering, t-SNE projections | Learned representations |
| **04. RNN** | Sequential data, hidden states, backprop through time | Text classification, sentiment analysis | Training curves, prediction samples | Sequence modeling basics |
| **05. LSTM** | Vanishing gradients, memory cells, gating mechanisms | Gate activations, cell state visualization | Architecture diagrams, attention to error patterns | Managing long-term dependencies |
| **06. Attention** | Query-Key-Value mechanism, attention scores | Custom attention layer implementation | Attention weight heatmaps, interpretability | Selective focus in sequences |
| **07. Self-Attention** | Scaled dot-product, token interactions, multi-head foundation | NumPy implementation from scratch | Attention matrices, token relationships | Building blocks for Transformers |

---

## Technologies Used

- **Python** — Core language
- **NumPy** — Numerical computation
- **Pandas** — Data manipulation
- **Matplotlib & Seaborn** — Visualization
- **Scikit-learn** — Classical ML and utilities
- **TensorFlow/Keras** — Deep learning framework
- **Gensim** — Word embeddings (Word2Vec)
- **Jupyter Notebook** — Interactive environment

---

## Visual Highlights

### TF-IDF Heatmap
Shows the importance of terms across documents, highlighting how different keywords distinguish document topics.

![TF-IDF Heatmap Placeholder](https://via.placeholder.com/600x400?text=TF-IDF+Term+Importance+Heatmap)

*Visualization from `01-tf-idf/notebook.ipynb` — displays term frequency across documents*

---

### Word Embedding PCA Projection
Demonstrates how learned embeddings cluster semantically similar words in 2D space, revealing semantic structure.

![Word Embedding PCA Placeholder](https://via.placeholder.com/600x400?text=Word+Embeddings+PCA+Projection)

*Visualization from `03-word-embeddings/notebook.ipynb` — semantic relationships in vector space*

---

### RNN Training Curves
Illustrates model convergence during training and validation on sentiment classification tasks.

![RNN Training Curves Placeholder](https://via.placeholder.com/600x400?text=RNN+Training+Convergence+Curves)

*Visualization from `04-rnn/notebook.ipynb` — loss and accuracy over epochs*

---

### LSTM Cell Architecture
Visualizes the flow of information through forget, input, and output gates, explaining memory dynamics.

![LSTM Cell Architecture Placeholder](https://via.placeholder.com/600x400?text=LSTM+Cell+Gate+Architecture)

*Diagram from `05-lstm/notebook.ipynb` — forget gate, input gate, output gate flow*

---

### Attention Heatmap
Displays how attention weights distribute across input tokens, revealing which words the model focuses on during prediction.

![Attention Heatmap Placeholder](https://via.placeholder.com/600x400?text=Attention+Weight+Distribution+Heatmap)

*Visualization from `06-attention-mechanisms/notebook.ipynb` — interpretability of attention focus*

---

### Self-Attention Matrix
Visualizes pairwise token interactions and learned relationships in the self-attention mechanism, showing token-to-token attention patterns.

![Self-Attention Matrix Placeholder](https://via.placeholder.com/600x400?text=Self-Attention+Token+Interaction+Matrix)

*Visualization from `07-self-attention/notebook.ipynb` — token relationships in sequence*

---

## Key Skills Acquired

- [x] Sparse text representation (Bag of Words, TF-IDF)
- [x] Feature engineering for NLP
- [x] Semantic search and similarity
- [x] Word embeddings and dense vectors
- [x] Sequence modeling with RNNs
- [x] Sentiment analysis and text classification
- [x] Handling vanishing gradients (LSTM)
- [x] Attention mechanisms and interpretability
- [x] Self-attention and multi-head concepts
- [x] Foundation for Transformer architectures

---

## Repository Philosophy

This folder is intentionally structured as a **progressive learning path**, not a collection of isolated notebooks. Each technique builds on the previous one, creating a narrative:

- **Why** was each technique introduced?
- **What problem** does it solve?
- **What are its limitations?**
- **Why** did we need the next technique?

For example, TF-IDF works well for exact keyword matching but lacks semantic understanding. Word embeddings capture meaning but don't capture context. RNNs model sequences but struggle with long-term dependencies. LSTMs fix this but remain computationally expensive. Attention mechanisms allow selective focus. Self-attention enables parallelization, paving the way for Transformers.

The notebooks reflect this progression through incremental complexity, building intuition before diving into implementation.

---

## Next Steps

The next section of this repository introduces **Transformer architectures** and **Large Language Models**, where Self-Attention becomes the core building block of modern NLP systems. Self-Attention scales from processing individual documents to understanding entire sequences in parallel, forming the foundation of BERT, GPT, and contemporary language models.

---

## Getting Started

1. Start with **01-tf-idf** to understand text as vectors
2. Progress sequentially through each notebook
3. Run cells interactively and experiment with modifications
4. Pay attention to the visualizations—they encode key insights
5. Review each notebook's limitations section to understand motivation for the next technique

Each notebook is self-contained and includes all necessary data and explanations to run independently, but the sequence maximizes learning retention.

---

## License

This repository is part of a public learning portfolio. Educational use is encouraged.
