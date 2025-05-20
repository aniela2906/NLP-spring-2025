# NLP Spring 2025 – Named Entity Recognition on Song Lyrics

This repository contains the full pipeline for a research project on adapting NER models to song lyrics using fine-tuning and continual learning techniques.

##  Repository Structure

| Path | Description |
|------|-------------|
| `3genres/` | Contains merged datasets and predictions for the 3-genre model (pop, country, rap/hip-hop). Used in fine-tuning and continual learning. |
| `baseline/` | Contains baseline prediction on EWT, and baseline prediction on `lyrics_test.iob2`.  |
| `country/` | Country-specific datasets, predictions(/country/predictions/), and orgnial songs & lyrics (/country/songs+orgnial_lyrics). Includes also manual annotations (/country/datasets/manual/) and pseudo-labeled data (/country/datasets/ country_labeled_no_2000). |
| `datasets_original/` | English Web Treebank (EWT) data used for initial baseline and DAPT pretraining. |
| `ner_DAPT_model/` | Contains baseline prediction on EWT, and baseline prediction on `lyrics_test.iob2`. |
| `pop/` | Pop-specific datasets, predictions(/pop/predictions/), and orgnial songs & lyrics (/pop/songs+orgnial_lyrics). Includes also manual annotations (/pop/datasets/manual/) and pseudo-labeled data (/pop/datasets/ pop_labeled_no_2000). |
| `rap-hip-hop/` | Hip-hop-specific datasets, predictions(/rap-hip-hop/predictions/), and orgnial songs & lyrics (/rap-hip-hop/songs+orgnial_lyrics). Includes also manual annotations (/rap-hip-hop/datasets/manual/) and pseudo-labeled data (/rap-hip-hop/datasets/ rap-hip-hop_labeled_no_2000). |
| `test/`| Contains the lyrics test set with and without labels. |
| `detailed_results.pdf`| Combined results of all models. |
| `main_code.ipynb` | Main notebook containing baseline model, model training code, pseudo-label generation, continuous learning, predictions, and error analysis. |
| `README.md` | This file. Describes the full repository structure, usage guide, setup instructions, and steps for running the baseline and DAPT models.|
| `raport.pdf` | Project raport.|
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
## Initial Baseline Model : 

The code is included in the first cell in the `main_code.ipynb`. 
The prediction on `lyrics_test.iob2` is saved to : 
```bash
/baseline/baseline_predictions.iob2.  
```
    
Evaluation results on the **English Web Treebank (EWT) test set**:   
    
recall:     0.8591160220994475    
precision:  0.8536139066788655    
span-f1:    0.8563561266636073    
 


## How to get DAPT model :
 1. Install repository:
  ```bash
  git clone https://github.com/huggingface/transformers.git
  ```
  
2. Navigate to tranformers:
 ```bash
 cd transformers
 ```
  
3. Install requirements:
```bash
 pip install -r examples/pytorch/language-modeling/requirements.txt
  ```
  
4. Install editable mode:
 ```bash
   pip install -e 
  ```
5.Run run_mlm.py:
```bash
python examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path deepset/roberta-base-squad2 \
    --train_file /PATH TO LYRICS DATASET /all_lyrics.txt \
    --do_train \
    --output_dir /PATH TO SAVE THE MODEL/ner_DAPT_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --logging_steps 100 \
    --save_steps 500 \
    --max_seq_length 128 \
    --line_by_line True
 ```

The prediction on `lyrics_test.iob2` is saved to : 
```bash
/ner_DAPT_model/predictions_ner_DAPT_model_lyrics.iob2.  
```
    
   
### RESULTS DAPT MODEL:
1. Results on EWT test data:
     
 ```bash
 python span_f1.py en_ewt-ud-test-unmasked.iob2 ../ner_DAPT_model/predictions_ner_DAPT_model_ewt.iob2
  ```
recall:    0.8492647058823529  
precision: 0.8676056338028169  
span-f1:   0.8583372039015327  
  
unlabeled  
ul_recall:    0.8943014705882353  
ul_precision: 0.9136150234741784  
ul_span-f1:   0.903855085926614  
  
loose (partial overlap with same label)  
l_recall:    0.8639705882352942  
l_precision: 0.8807511737089202  
l_span-f1:   0.8722801838503713  

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

### 3. Continuous Learning
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
