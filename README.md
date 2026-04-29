<p align="center">
  <img src="https://img.shields.io/badge/🏥-Medical%20NER-blue?style=for-the-badge" alt="Medical NER"/>
</p>

<h1 align="center">🏥 Medical Symptoms Extraction Named Entity Recognition (NER)</h1>

<p align="center">
  <b>Extract symptoms, diseases, drugs & clinical entities from biomedical text using Transformers</b>
</p>

<p align="center">
  <a href="https://www.kaggle.com/code/ahmedaboenaba/medical-ner">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"/>
  </a>
  &nbsp;&nbsp;
  <img src="https://img.shields.io/badge/python-3.12-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/🤗%20Transformers-5.0-orange?style=flat-square" alt="Transformers"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"/>
</p>

---

## 📌 Overview

This project builds an **end-to-end pipeline** for extracting medical entities — such as **chemicals**, **drugs**, and **diseases** — from PubMed biomedical abstracts using state-of-the-art Transformer models.

We fine-tune and compare **three model variants**:

| # | Model | Encoder | Classification Head | Role |
|---|-------|---------|---------------------|------|
| A | **BERT Baseline** | `bert-base-uncased` | Linear + Softmax | Baseline |
| B | **BioBERT** | `dmis-lab/biobert-v1.1` | Linear + Softmax | Main Model |
| C | **BioBERT + CRF** | `dmis-lab/biobert-v1.1` | Linear + CRF | Ablation Study |

> 💡 **Why CRF?** — The CRF (Conditional Random Field) layer enforces valid BIO transition constraints (e.g., `I-Disease` cannot follow `B-Chemical`), reducing boundary errors.

---

## 📂 Project Structure

```
Medical-symptom-extraction-NER-/
│
├── Medical NER.ipynb    # 📓 Complete pipeline notebook (data → deployment)
├── README.md            # 📖 You are here
└── .gitignore
```

---

## 🧬 Datasets

We use two standard biomedical NER datasets from HuggingFace:

| Dataset | Source | Abstracts | Entity Types | Label Schema |
|---------|--------|-----------|--------------|--------------|
| **BC5CDR** | BioCreative V | ~1,500 PubMed | Chemical, Disease | BIO (5 tags) |
| **NCBI Disease** | NCBI | 793 PubMed | Disease | BIO |

### Label Mapping (BC5CDR)

```
0 → O            (Outside any entity)
1 → B-Chemical   (Beginning of a Chemical entity)
2 → I-Chemical   (Inside a Chemical entity)
3 → B-Disease    (Beginning of a Disease entity)
4 → I-Disease    (Inside a Disease entity)
```

---

## ⚙️ Pipeline at a Glance

The notebook follows a clear, step-by-step workflow:

```
Section 0   →   Setup & Installation
Section 1   →   Data Loading & Exploration (EDA)
Section 2   →   Preprocessing & Tokenization (WordPiece alignment)
Section 3   →   Model Architecture (BERT / BioBERT / BioBERT+CRF)
Section 4   →   Training Loop (HuggingFace Trainer)
Section 5   →   Evaluation & Metrics (seqeval span-level F1)
Section 6   →   Error Analysis
Section 7   →   Cross-Dataset Transfer (BC5CDR → NCBI Disease)
Section 8   →   Interactive Demo (Gradio)
Section 9   →   Deployment (HuggingFace Hub)
Section 10  →   Thesis Summary & References
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR-USERNAME/Medical-symptom-extraction-NER-.git
cd Medical-symptom-extraction-NER-
```

### 2. Install dependencies

```bash
pip install transformers datasets seqeval evaluate \
    pytorch-crf wandb gradio accelerate \
    matplotlib seaborn pandas numpy scikit-learn
```

### 3. Run the notebook

Open `Medical NER.ipynb` in **Jupyter**, **Google Colab**, or **Kaggle** and run all cells. A GPU (e.g., Tesla T4) is recommended for training.

### 4. Try it on Kaggle 🎯

Click the badge below to run the notebook directly on Kaggle:

<p align="center">
  <a href="https://www.kaggle.com/code/ahmedaboenaba/medical-ner">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" width="200"/>
  </a>
</p>

---

## 🧠 Key Findings

| Finding | Detail |
|---------|--------|
| 🏆 **Domain pre-training matters** | BioBERT significantly outperforms BERT-base on both Chemical and Disease recognition |
| 🔗 **CRF adds structural consistency** | CRF reduces boundary errors by enforcing valid BIO transitions |
| 🔄 **Transfer learning works** | Zero-shot transfer from BC5CDR → NCBI Disease shows strong results |
| 🔍 **Error analysis reveals** | Boundary errors dominate, especially for multi-word chemical names |

---

## 🔬 How It Works

### Tokenization & Label Alignment

When BERT tokenizes a word like `"Naloxone"` into sub-tokens `["Na", "##lo", "##xon", "##e"]`, only the **first sub-token** gets the NER label. All subsequent sub-tokens receive `-100` (ignored by PyTorch's loss function):

```
Token         Label ID    Label Name       Status
─────         ────────    ──────────       ──────
[CLS]         -100        ---              IGNORED
Na             1          B-Chemical       ASSIGNED  ◄
##lo          -100        ---              IGNORED
##xon         -100        ---              IGNORED
##e           -100        ---              IGNORED
reverses       0          O                ASSIGNED
...
[SEP]         -100        ---              IGNORED
```

### Model Architecture (BioBERT + CRF)

```
Input Text
    ↓
BioBERT Encoder (pre-trained on PubMed)
    ↓
Dropout (0.1)
    ↓
Linear Classifier (hidden_size → num_labels)
    ↓
CRF Layer (enforces valid BIO transitions)
    ↓
Predicted Entity Tags
```

---

## 📊 Example Output

Given the input:
> *"Patient has chest pain and was given aspirin 500mg."*

The model extracts:

| Token | Predicted Label |
|-------|----------------|
| chest | `B-Disease` |
| pain | `I-Disease` |
| aspirin | `B-Chemical` |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Deep Learning | PyTorch |
| NLP Framework | 🤗 HuggingFace Transformers |
| Pre-trained Model | BioBERT v1.1 |
| Structured Prediction | pytorch-crf |
| Evaluation | seqeval |
| Visualization | Matplotlib, Seaborn |
| Interactive Demo | Gradio |
| Deployment | HuggingFace Hub |
| GPU | Tesla T4 (Kaggle/Colab) |

---

## 📚 References

1. J. Lee et al., *"BioBERT: a pre-trained biomedical language representation model for biomedical text mining,"* Bioinformatics, 2020.
2. Y. Gu et al., *"Domain-Specific Language Model Pretraining for Biomedical NLP,"* ACM THCA, 2022.
3. J. Li et al., *"BioCreative V CDR task corpus,"* Database, 2016.
4. R. I. Doğan et al., *"NCBI disease corpus,"* J. Biomedical Informatics, 2014.
5. J. Devlin et al., *"BERT: Pre-training of Deep Bidirectional Transformers,"* NAACL-HLT, 2019.
6. T. Wolf et al., *"Transformers: State-of-the-Art NLP,"* EMNLP, 2020.

---

## 👨‍💻 Author

**Ahmed Jaber** — Master's Thesis in NLP

---

## 📄 License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

---

<p align="center">
  <i>⭐ If you found this project useful, please consider giving it a star!</i>
</p>
