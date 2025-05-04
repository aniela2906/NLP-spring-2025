# NLP Spring 2025 – Named Entity Recognition on Song Lyrics

This repository contains the full pipeline for a research project on adapting NER models to song lyrics using fine-tuning and continual learning techniques.

## 📁 Repository Structure

| Path | Description |
|------|-------------|
| `3genres/` | Contains merged datasets and predictions for the 3-genre model (pop, country, rap/hip-hop). Used in fine-tuning and continual learning. |
| `pop/` | Pop-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `country/` | Country-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `rap-hip-hop/` | Hip-hop-specific datasets and predictions. Includes manual annotations and pseudo-labeled data. |
| `datasets_original/` | English Web Treebank (EWT) data used for initial baseline and DAPT pretraining. |
| `model_trainings_predictions_statistics.ipynb` | Main notebook containing model training code, pseudo-label generation, continuous learning, predictions, and error analysis. |
| `README.md` | This file. Full explanation of project goals, structure, and usage. |

