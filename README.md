# NLP Spring 2025 – Named Entity Recognition on Song Lyrics

This repository contains the full pipeline for a research project on adapting NER models to song lyrics using fine-tuning and continual learning techniques.

##  Repository Structure

| Path | Description |
|------|-------------|
| `3genres/` | Contains merged datasets and predictions for the 3-genre model (pop, country, rap/hip-hop). Used in fine-tuning and continual learning. |
| `baseline/` | Contains baseline prediction on EWT, and baseline prediction on `lyrics_test.iob2`.  |
| `country/` | Country-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `datasets_original/` | English Web Treebank (EWT) data used for initial baseline and DAPT pretraining. |
| `ner_DAPT_model/` | Contains baseline prediction on EWT, and baseline prediction on `lyrics_test.iob2`. |
| `pop/` | Pop-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `rap-hip-hop/` | Hip-hop-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `test/`| Contains the lyrics test set with and without labels. |
| `detailed_results.pdf`| Combined results of all models. |
| `main_code.ipynb` | Main notebook containing model training code, pseudo-label generation, continuous learning, predictions, and error analysis. |
| `README.md` | This file. Full explanation of project goals, structure, and usage. |
| `requirements.txt` | File containing all necessary libraries to run your NER training and evaluation code. | 


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
## Baseline model: 

The code is included in the first cell in the `main_code.ipynb`. 
Also its prediction on `lyrics_test.iob2` is saved to : 
```bash
/baseline/baseline_predictions.iob2.  
```
To evaluate the baseline on the EWT dataset, run the following command from the datasets_orginal/ directory:  

```bash
python span_f1.py en_ewt-ud-test-masked.iob2 ../baseline/baseline_predictions.iob2  
```
  
Assuming you downloaded and organized the repository as follows:  

NLP-spring-2025/  
├── baseline/  
│   └── baseline_predictions.iob2  
├── datasets_orginal/  
│   ├── span_f1.py  
│   └── en_ewt-ud-test-masked.iob2  
├──test  
│    └──lyrics_test.iob2
    
You should see the following evaluation results on the **English Web Treebank (EWT) test set**:   
    
recall:     0.8591160220994475    
precision:  0.8536139066788655    
slot-f1:    0.8563561266636073    


In the detailed_results.pdf it is reffered as **EWT (basic)**.  


## DAPT model :
 1. Install dependencies (if not yet):
  ```bash
  pip install transformers datasets
  ```

2. File with all lyrics is saved in /ner_DAPT_model/all_lyrics.txt

3. To get `run_mlm.py` for MLM pretraining, you'll need to clone the Hugging Face Transformers repository from GitHub.

in terminal clone the repo:
 ```bash
  git clone https://github.com/huggingface/transformers.git
cd transformers
  ```

4. After cloning, the script is located in transformers/examples/pytorch/language-modeling/run_mlm.py

### RESULTS DAPT MODEL:
1. Results on EWT test data:
     
 ```bash
 python span_f1.py en_ewt-ud-test-unmasked.iob2 ../ner_DAPT_model/predictions_ner_DAPT_model_ewt.iob2
  ```
recall:    0.8492647058823529  
precision: 0.8676056338028169  
slot-f1:   0.8583372039015327  
  
unlabeled  
ul_recall:    0.8943014705882353  
ul_precision: 0.9136150234741784  
ul_slot-f1:   0.903855085926614  
  
loose (partial overlap with same label)  
l_recall:    0.8639705882352942  
l_precision: 0.8807511737089202  
l_slot-f1:   0.8722801838503713  

Also its prediction on `lyrics_test.iob2` is saved to : 
```bash
/ner_DAPT_model/predictions_ner_DAPT_model_lyrics.iob2.  
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
