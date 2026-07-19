# LLM 
Learning Large Language Models from Fundamentals to Production

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]() [![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)]() [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)]()

> A hands-on, portfolio-style journey into Large Language Models. This repository contains concise, reproducible notebooks and projects that implement core LLM concepts, from data pipelines and transformer internals to fine‑tuning, quantization, evaluation, and production-grade serving.

---

## 🚀 Quick Links
- Notebook-first examples — follow the roadmap notebooks to reproduce experiments.
- Visual explanations — diagrams and plots to explain model behavior.
- Production notes — deployment, inference optimization, and security.

---

## Repository Structure
```text
LLM-Engineering/
├─ notebooks/                # Reproducible Jupyter notebooks (experiments & tutorials)
│  ├─ 01-data-prep.ipynb
│  ├─ 02-transformer-from-scratch.ipynb
│  ├─ 03-autoregressive.ipynb
│  ├─ 04-small-models.ipynb
│  ├─ 05-peft-lora.ipynb
│  ├─ 06-quantization.ipynb
│  ├─ 07-evaluation.ipynb
│  ├─ 08-rlhf-dpo.ipynb
│  ├─ 09-serving-inference.ipynb
│  └─ 10-api-gateway-security.ipynb
├─ src/                      # Implementation modules, utilities, and training scripts
├─ viz/                      # Static diagrams & rendered plots
├─ deployments/              # Docker + FastAPI examples, helm charts (proofs-of-concept)
├─ docs/                     # Short guides, diagrams, references
├─ data/                     # Small sample datasets and preprocessing recipes
└─ README.md
```

---

## Topics Covered
| Topic | Description | Skills Learned |
|---|---:|---|
| Data Preprocessing | Tokenization, cleaning, batching, and dataset pipelines | Text cleaning, tokenization, dataset sharding |
| Transformer Architecture | Build simplified transformer encoder/decoder blocks | Attention, positional encodings, backprop |
| Temperature & Autoregression | Sampling strategies, temperature, top-k/top-p | Sampling math, logits manipulation |
| Small Language Models | Train and debug small LMs on toy corpora | Training loop, loss analysis, checkpoints |
| PEFT & LoRA | Parameter-efficient fine-tuning recipes | LoRA adapters, low-rank updates, experiment design |
| Quantization | 8-bit/4-bit quantization workflows & evaluation | BitsAndBytes usage, accuracy-vs-size tradeoffs |
| Evaluation Metrics | Perplexity, BLEU, ROUGE, embedding-based metrics | Metric implementation, A/B comparisons |
| RLHF & DPO | Alignment through preference models and optimization | Reward modeling, policy updates, DPO concepts |
| Serving & Inference | Batch vs realtime, latency, memory management | TorchScript, memory mapping, batching strategies |
| API Gateway | End-to-end API design and rate limiting | FastAPI, authentication, quotas, telemetry |
| LLM Security | Adversarial prompts, data leaks, supply chain risks | Red-teaming basics, input sanitization, secrets hygiene |

---

> 💡 **Highlight:** Each topic folder/notebook pairs a short conceptual summary with runnable code, visual diagnostics, and "how I'd productionize this" notes.

---

## What I Learned (Practical Skills)
- Text preprocessing and dataset engineering  
- Tokenization and subword strategies  
- Transformer internals and attention math  
- Autoregressive generation and sampling controls  
- Fine-tuning strategies and LoRA/PEFT workflows  
- Quantization and inference memory optimizations  
- Model evaluation and metric design  
- Scaling inference: batching, caching, and sharding  
- Deploying with FastAPI + Docker, API design patterns  
- Security best practices for LLMs (red-teaming, secrets handling)

---

## Technologies
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FF6E1A?logo=transformers&logoColor=white) ![HuggingFace](https://img.shields.io/badge/HuggingFace-FF6C37?logo=huggingface&logoColor=white) ![PEFT](https://img.shields.io/badge/PEFT-Adapter-blue) ![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-quant-orange)

![OpenAI](https://img.shields.io/badge/OpenAI-000000?logo=openai&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)

---

## Repository Highlights
- Step-by-step implementations that mirror a real engineering workflow  
- Practical experiments with reproducible notebooks  
- Visual explanations to build intuition (plots & diagrams)  
- Hands-on notebooks where every claim is demonstrated by code  
- Notes on production-ready considerations (cost, latency, security)

---

### Learning Philosophy
I learn by building: each concept is implemented from first principles, tested with small experiments, then connected to library-based solutions (e.g., Hugging Face, BitsAndBytes). The goal is deep understanding and transferable engineering practice rather than rote notes.

---

## Future Additions (Roadmap)
- [ ] RAG (Retrieval-Augmented Generation) pipelines  
- [ ] Vector databases (FAISS / Weaviate / Milvus)  
- [ ] AI Agents & tool use examples  
- [ ] Model Card & MCP (Model Cards + Model Governance)  
- [ ] Function calling patterns and safe schemas  
- [ ] Multi-agent systems experiments  
- [ ] Full fine-tuning projects (end-to-end)  
- [ ] LLMOps: CI, monitoring, and infra-as-code  
- [ ] Evaluation pipelines + AB testing harness

---

## References (select)
<details>
<summary>Books, papers & docs</summary>

- "Attention Is All You Need" — Vaswani et al.  
- "Scaling Laws for Neural Language Models" — Kaplan et al.  
- Hugging Face Transformers Docs — https://huggingface.co/docs/transformers  
- BitsAndBytes quantization guide — repository docs  
- OpenAI Safety & Best Practices pages

</details>

---

## How to Use
<details>
<summary>Run locally (quick start)</summary>

1. Clone the repo  
2. Create a virtualenv, pip install -r requirements.txt  
3. Open notebooks/01-data-prep.ipynb and run cells top-to-bottom  
4. See deployments/ for Docker + FastAPI examples

</details>

---

If you'd like, I can:
- Convert this into the actual README file in a repository you name, or
- Generate a ready-to-run requirements.txt and minimal Dockerfile to accompany the notebooks.
