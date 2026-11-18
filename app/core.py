import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import gradio as gr

# Global configurations for the project
MODEL_NAME = "distilbert-base-uncased"
categories = ["Pricing", "Usability", "Features", "Support"]
id2label = {i: label for i, label in enumerate(categories)}
label2id = {label: i for i, label in enumerate(categories)}

# --- Milestone 1: Define 3-5 feedback categories and create a small dataset of example comments with labels.
def milestone_1():
    # Dataset containing customer comments mapped to categories
    raw_dataset = [
        # Pricing
        ("The subscription fee is way too expensive for what it offers.", "Pricing"),
        ("I love the weekly discounts and fair price points.", "Pricing"),
        ("It costs too much money to upgrade my current plan.", "Pricing"),
        ("Very affordable rates compared to competitors.", "Pricing"),
        # Usability
        ("The interface is clean and super easy to navigate around.", "Usability"),
        ("I can't find where the settings button is, very confusing UI.", "Usability"),
        ("Extremely user-friendly design and smooth transitions.", "Usability"),
        ("The mobile app crashes constantly and is hard to use.", "Usability"),
        # Features
        ("I wish there was an automatic dark mode option available.", "Features"),
        ("The export to CSV feature works flawlessly!", "Features"),
        ("Please add integration with third-party calendars.", "Features"),
        ("Missing core features like collaborative sharing.", "Features"),
        # Support
        ("The customer service agent resolved my ticket in minutes!", "Support"),
        ("Nobody replied to my email support query for 3 days.", "Support"),
        ("Incredible live chat experience, highly recommend.", "Support"),
        ("Help desk documentation is completely outdated.", "Support"),
    ]
    
    texts = [item[0] for item in raw_dataset]
    labels = [label2id[item[1]] for item in raw_dataset]
    
    # Split into Train and Validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    return X_train, X_val, y_train, y_val

# --- Milestone 2: Extend / Load Tokenizer and encode dataset.
class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def milestone_2(X_train, X_val, y_train, y_val):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize text data
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)
    
    # Wrap into PyTorch datasets
    train_dataset = FeedbackDataset(train_encodings, y_train)
    val_dataset = FeedbackDataset(val_encodings, y_val)
    
    return tokenizer, train_dataset, val_dataset

# --- Milestone 3: Fine-tune the classification model using the Hugging Face Trainer.
def milestone_3(train_dataset, val_dataset):
    # Initialize classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id
    )
    
    # Define training arguments
    # Changed 'evaluation_strategy' to 'eval_strategy' due to HF v4.41.0+ changes.
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        logging_steps=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        disable_tqdm=True,
        report_to="none"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train and return
    trainer.train()
    return model
