import torch
import os
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import RobertaForTokenClassification
from tqdm import tqdm

# tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

# labels
tag2idx = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6}
idx2tag = {v: k for k, v in tag2idx.items()}

# dataset class
class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, tag2idx, max_len=50):
        self.sentences, self.labels, self.raw_data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_len = max_len

    def load_data(self, file_path):
        sentences, labels, raw_data = [], [], []
        sentence, label, sentence_data = [], [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                sentence_data.append(line)
                if line.startswith('#') or not line:
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        raw_data.append(sentence_data)
                        sentence, label, sentence_data = [], [], []
                else:
                    parts = line.split('\t')
                    if len(parts) > 2:
                        sentence.append(parts[1])  # word
                        label.append(parts[2])  # tag
        return sentences, labels, raw_data

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        labels = self.labels[idx]

        encodings = self.tokenizer(words, truncation=True, padding="max_length", max_length=self.max_len, is_split_into_words=True, return_tensors="pt")
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        label_ids = [-100] * len(input_ids)
        word_ids = encodings.word_ids()
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None:
                label_ids[i] = self.tag2idx.get(labels[word_idx], self.tag2idx['O'])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids)
        }

# files
data_files = {
    "train": "/work/project/en_ewt-ud-train.iob2",
    "dev": "/work/project/en_ewt-ud-dev.iob2",
    "test": "/work/project/en_ewt-ud-test-masked.iob2"
}

datasets = {split: NERDataset(data_files[split], tokenizer, tag2idx) for split in data_files}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dropout---> to prevent overfitting & memorization
model = RobertaForTokenClassification.from_pretrained(
    "roberta-base",
    num_labels=len(tag2idx),
    hidden_dropout_prob=0.3,  
    attention_probs_dropout_prob=0.3,
)
model.to(device)

# parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  
    metric_for_best_model="loss",
    greater_is_better=False,  
    learning_rate=3e-05,  
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=64,
    num_train_epochs=5,  
    weight_decay=0.01,  
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,  
    fp16=True,  
)

# train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["dev"]
)


trainer.train()


model.eval()
output_path = "/work/project/predictions.iob2"
with open(output_path, "w", encoding="utf-8") as f:
    with torch.no_grad():
        for i, sentence_data in enumerate(datasets["test"].raw_data):
            for line in sentence_data:
                if line.startswith("#"):
                    f.write(line + "\n")

            words = datasets["test"].sentences[i]
            encodings = tokenizer(words, truncation=True, padding="max_length", max_length=128, is_split_into_words=True, return_tensors="pt")
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            pred_labels = torch.argmax(outputs.logits, dim=2).cpu().numpy()[0]

            word_ids = encodings.word_ids()
            valid_predictions = [idx2tag[pred_labels[i]] for i, word_idx in enumerate(word_ids) if word_idx is not None]

            for j, (word, label) in enumerate(zip(words, valid_predictions)):
                f.write(f"{j+1}\t{word}\t{label}\t-\t-\n")

            f.write("\n")

print(f"\n Predictions saved to: {output_path}")