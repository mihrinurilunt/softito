# NLP Foundations

A progressive learning journey through Natural Language Processing, tracing the evolution of techniques from sparse lexical representations to modern attention-based architectures. Each notebook is fully executable, well-documented, and implements concepts from first principles.

---

## Learning Journey

```
┌─────────────────────┐
│   TF-IDF            │  Sparse Vectors
│  (Baseline)         │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Word Vectors       │  Dense Representations
│  (Semantic Space)   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Word Embeddings     │  Learned Representations
│ (Word2Vec, GloVe)   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│      RNN            │  Sequence Modeling
│  (Hidden States)    │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│      LSTM           │  Long-term Dependencies
│  (Memory Cells)     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Attention          │  Selective Focus
│ (Query-Key-Value)   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Self-Attention      │  Token Interactions
│ (Scaled Dot-Prod)   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Transformers       │  Parallel Processing
│   (Next Section)    │
└─────────────────────┘
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
│   └── notebook.ipynb
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

## Visual Architecture Diagrams

### TF-IDF: Sparse Vector Representation

```
Document 1: "machine learning"  ──┐
Document 2: "deep learning"     ──├──> TF-IDF Vectorizer ──> [0.5, 0.3, 0.0, ...]
Document 3: "neural networks"   ──┘                            Sparse Vector
                                                               (high-dim, sparse)
```

**Key Insight**: Each dimension represents a unique word; most values are zero.

---

### Word Embeddings: Dense Vector Space

```
Similarity Clustering in 2D Space:
┌────────────────────────────────────────┐
│                                        │
│  good ●                                │
│         ● excellent                    │
│            ● great                     │
│                                        │
│                                        │
│  bad ●                                 │
│      ● poor                            │
│         ● terrible                     │
│                                        │
└────────────────────────────────────────┘
     Semantic Similarity ────>
```

**Key Insight**: Similar words cluster together; geometry encodes meaning.

---

### RNN: Sequence Processing with Hidden State

```
Input Sequence: "I love NLP"

    x₁: "I"     x₂: "love"    x₃: "NLP"
      ↓             ↓             ↓
    [RNN] ──> h₁  [RNN] ──> h₂  [RNN] ──> h₃
      ↑             ↑             ↑
    h₀=0          h₁            h₂
    
Output: h₃ (final prediction)
```

**Key Insight**: Hidden state carries information from all previous steps.

---

### LSTM: Memory Cell with Gates

```
Input: xₜ
Current Hidden State: hₜ₋₁
Previous Cell State: Cₜ₋₁

        ┌─────────────────────────────┐
        │   LSTM Cell Architecture    │
        │                             │
xₜ ─────┤  ┌──────────┐              │
        │  │ Forget   │ ─ remove old │
hₜ₋₁ ───┤  │  Gate    │   info      │
        │  └──────────┘              │
Cₜ₋₁ ───┤                            │
        │  ┌──────────┐              │
        │  │  Input   │ ─ add new   │
        │  │  Gate    │   info      │
        │  └──────────┘              │
        │                            │
        │  ┌──────────┐              │
        │  │ Output   │ ─ select    │
        │  │  Gate    │   output    │
        │  └──────────┘              │
        │                             │
        └────────┬────────────────────┘
                 │
            ┌────┴─────┐
            ↓          ↓
           Cₜ          hₜ
        (Cell State) (Output)
```

**Key Insight**: Gates control what information flows through the cell.

---

### Attention Mechanism: Query-Key-Value

```
Input Sequence: [word₁, word₂, word₃, word₄]

Step 1: Create Q, K, V matrices
        ┌──────────┬──────────┬──────────┬──────────┐
Q matrix│   q₁     │   q₂     │   q₃     │   q₄     │
        └──────────┴──────────┴────────���─┴──────────┘
        ┌──────────┬──────────┬──────────┬──────────┐
K matrix│   k₁     │   k₂     │   k₃     │   k₄     │
        └──────────┴──────────┴──────────┴──────────┘
        ┌──────────┬──────────┬──────────┬──────────┐
V matrix│   v₁     │   v₂     │   v₃     │   v₄     │
        └──────────┴──────────┴──────────┴──────────┘

Step 2: Compute Attention Scores (Q · K^T)
        ┌─────┬─────┬─────┬─────┐
Scores  │0.9  │0.1  │0.0  │0.0  │
        ├─────┼─────┼─────┼─────┤
        │0.2  │0.7  │0.1  │0.0  │
        ├─────┼─────┼─────┼─────┤
        │0.1  │0.2  │0.6  │0.1  │
        ├─────┼─────┼─────┼─────┤
        │0.0  │0.1  │0.2  │0.7  │
        └─────┴─────┴─────┴─────┘
   (What each word attends to)

Step 3: Weighted Sum of Values
Output = Attention(Q,K,V) = softmax(Scores) · V
```

**Key Insight**: Attention weights show which inputs are relevant to each output.

---

### Self-Attention: Token-to-Token Relationships

```
Sentence: "The cat sat on the mat"

Self-Attention Matrix (showing which tokens interact):

        The  cat  sat  on  the  mat
    ┌────┬────┬────┬────┬────┬────┐
The │0.9 │0.1 │0.0 │0.0 │0.8 │0.0 │
    ├────┼────┼────┼────┼────┼────┤
cat │0.1 │0.9 │0.7 │0.0 │0.1 │0.1 │
    ├────┼────┼────┼────┼────┼────┤
sat │0.0 │0.6 │0.8 │0.9 │0.0 │0.0 │
    ├────┼────┼────┼────┼────┼────┤
on  │0.0 │0.0 │0.7 │0.9 │0.0 │0.8 │
    ├────┼────┼────┼────┼────┼────┤
the │0.7 │0.1 │0.0 │0.0 │0.9 │0.9 │
    ├────┼────┼────┼────┼────┼────┤
mat │0.0 │0.2 │0.0 │0.8 │0.8 │0.9 │
    └────┴────┴────┴────┴────┴────┘

Darker values = stronger attention connections
"sat" attends strongly to "cat" (subject)
"on" attends to "sat" (verb) and "mat" (object)
```

**Key Insight**: Each token learns to attend to relevant context tokens.

---

## Technologies Used

- **Python** — Core language
- **NumPy** — Numerical computation and linear algebra
- **Pandas** — Data manipulation and analysis
- **Matplotlib & Seaborn** — Visualization and plotting
- **Scikit-learn** — Classical ML algorithms and utilities
- **TensorFlow/Keras** — Deep learning framework
- **Gensim** — Word embeddings (Word2Vec training)
- **Jupyter Notebook** — Interactive environment

---

## Key Skills Acquired

- [x] Sparse text representation (Bag of Words, TF-IDF)
- [x] Feature engineering for NLP tasks
- [x] Semantic search and similarity computation
- [x] Word embeddings and dense vector spaces
- [x] Sequence modeling with RNNs
- [x] Sentiment analysis and text classification
- [x] Handling vanishing/exploding gradients (LSTM)
- [x] Attention mechanisms and interpretability
- [x] Self-attention and scaled dot-product attention
- [x] Foundation for Transformer architectures

---

## Repository Philosophy

This folder is intentionally structured as a **progressive learning path**, not a collection of isolated notebooks. Each technique builds on the previous one, creating a narrative:

- **Why** was each technique introduced?
- **What problem** does it solve?
- **What are its limitations?**
- **Why** did we need the next technique?

For example:
- **TF-IDF** works well for exact keyword matching but lacks semantic understanding
- **Word Embeddings** capture meaning but don't capture context
- **RNNs** model sequences but struggle with long-term dependencies
- **LSTMs** fix vanishing gradients but remain computationally expensive
- **Attention** allows selective focus on relevant information
- **Self-Attention** enables parallelization, paving the way for Transformers

The notebooks reflect this progression through incremental complexity, building intuition before diving into implementation.

---

## Next Steps

The next section of this repository introduces **Transformer architectures** and **Large Language Models**, where Self-Attention becomes the core building block of modern NLP systems. Self-Attention scales from processing individual documents to understanding entire sequences in parallel, forming the foundation of BERT, GPT, T5, and contemporary language models.

---

## Getting Started

1. **Start with 01-tf-idf** — Understand text as vectors
2. **Progress sequentially** — Each notebook builds on the previous one
3. **Run interactively** — Modify cells and experiment
4. **Study visualizations** — They encode key insights
5. **Review limitations** — Understand why the next technique was needed

Each notebook is self-contained with all necessary data and explanations, but the sequence maximizes learning retention.

---

## License

This repository is part of a public learning portfolio. Educational use is encouraged.
