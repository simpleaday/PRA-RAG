# PRA-RAG

PRA-RAG is a framework for evaluating and defending against poisoning attacks in Retrieval-Augmented Generation (RAG) systems. It supports adversarial context injection, robust aggregation, and performance evaluation across multiple datasets.

## 🚀 Features

- Support for BEIR datasets (e.g., NQ, HotpotQA, MS MARCO)
- Adversarial attack generation and injection
- Robust RAG with diverse context selection
- Poisoning attack evaluation (ASR, PAD, ACC)
- Compatible with both open-source and API-based LLMs

---

## 📂 Project Structure
-main.py # Core evaluation and attack pipeline

-run.py # Script for batch experiments

-src/ # Models, attack, utils, prompts

-results/ # Evaluation outputs

-logs/ # Log files


---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- Transformers
- tqdm
- numpy

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🧪 Running Experiments

```bash
python main.py \
  --eval_dataset nq \
  --model_name gpt3.5 \
  --top_k 5 \
  --attack_method LM_targeted \
  --poison_ratio 0.2
```
