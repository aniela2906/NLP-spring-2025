# NLP Spring 2025 – Named Entity Recognition on Song Lyrics

This repository contains the full pipeline for a research project on adapting NER models to song lyrics using fine-tuning and continual learning techniques.

##  Repository Structure

| Path | Description |
|------|-------------|
| `3genres/` | Contains merged datasets and predictions for the 3-genre model (pop, country, rap/hip-hop). Used in fine-tuning and continual learning. |
| `pop/` | Pop-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `country/` | Country-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `rap-hip-hop/` | Hip-hop-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `datasets_original/` | English Web Treebank (EWT) data used for initial baseline and DAPT pretraining. |
| `model_trainings_predictions_statistics.ipynb` | Main notebook containing model training code, pseudo-label generation, continuous learning, predictions, and error analysis. |
| `requirements.txt` | file containing all necessary libraries to run your NER training and evaluation code (model_trainings_predictions_statistics.ipynb)
| `README.md` | This file. Full explanation of project goals, structure, and usage. |


##  Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/aniela2906/NLP-spring-2025.git
cd NLP-spring-2025
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
Use the provided requirements.txt file:
```bash
pip install -r requirements.txt
```

##  How to run: `main_code.ipynb`

Open the notebook and follow the code blocks in order. The notebook is structured in the following steps:

##  Note on Paths


In the notebook, you will find code cells like:

```python
# ------------------------------------------------------------------------------------------------------
BASE_MODEL_PATH = "/path/ner_DAPT_model"             # path to DAPT NER model
NEW_TRAIN_FILE  = "/path/pop-manual-1000-train.iob2" # path to training dataset
OUTPUT_MODEL_PATH = "/path/ner_DAPT_model_finetuned_on_pop"  # output model path
# ------------------------------------------------------------------------------------------------------
```
These paths are placeholders. You should replace them with the actual locations.

### All places where you need to adjust file paths are clearly marked in the code with:
```python
# ------------------------------------------------------------------------------------------------------
```

### 1. Fine-Tuning the DAPT NER Model
- Fine-tune a domain-adapted RoBERTa model on 1000 manually labeled song lyrics.
- Choose from: `pop`, `country`, `rap-hip-hop`, or a merged dataset (`3genres`).

### 2. Generating Pseudo-Labels
- Use the fine-tuned model to predict labels on additional unlabeled lyrics.
- This creates pseudo-labeled training data for self-training (continual learning).

### 3. Continual Learning
- Fine-tune again using both manual and pseudo-labeled data.
- This step improves generalization and robustness for lyrics-based NER.

### 4. Evaluation
- Predict labels on a held-out test set (`lyrics_test.iob2`)
- Analyze model performance using:
  - Token-level confusion matrix
  - Error type breakdown: `correct`, `missed`, `spurious`, `wrong_label`

---

##  Output Files

Models and predictions are saved in genre-specific folders:

- `ner_DAPT_model_finetuned_on_<genre>`
- `ner_DAPT_model_cont_on_<genre>`
- `predictions_<genre>.iob2`
- `predictions_continuous_learning_<genre>.iob2`
