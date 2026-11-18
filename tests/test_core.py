# milestone_1.py
def milestone_1():
    # Define feedback categories
    categories = ["Pricing", "Functionality", "Support", "UI/UX"]

    # Example comments with labels (0: Pricing, 1: Functionality, 2: Support, 3: UI/UX)
    comments = [
        ("The price is too high for what I get.", 0),
        ("I can't find the feature I need.", 1),
        ("The support team didn't respond to my email.", 2),
        ("The app crashes when I try to save.", 1),
        ("The checkout process is confusing.", 3),
        ("I wish there was a mobile app.", 1),
        ("The subscription renewal email is unclear.", 0),
        ("The login page is not mobile-friendly.", 3),
        ("I had to wait 3 days for a response.", 2),
        ("The dashboard is hard to navigate.", 3),
        ("The free tier doesn't include essential tools.", 0),
        ("The API documentation is outdated.", 1),
        ("The payment method options are limited.", 0),
        ("The app freezes when opening large files.", 1),
        ("I couldn't find my account settings.", 3),
        ("The support chat is slow to respond.", 2)
    ]

    # Split into train (75%) and validation (25%)
    split_idx = int(0.75 * len(comments))
    X_train, y_train = zip(*comments[:split_idx])
    X_val, y_val = zip(*comments[split_idx:])

    return list(X_train), list(X_val), list(y_train), list(y_val)

# milestone_2.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
import torch

class FeedbackDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=128):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            comment,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def milestone_2(X_train, X_val, y_train, y_val):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add custom category tokens
    category_tokens = ["[Pricing]", "[Functionality]", "[Support]", "[UI/UX]"]
    tokenizer.add_tokens(category_tokens)

    train_dataset = FeedbackDataset(X_train, y_train, tokenizer)
    val_dataset = FeedbackDataset(X_val, y_val, tokenizer)

    return tokenizer, train_dataset, val_dataset

# milestone_3.py
from transformers import Trainer, TrainingArguments
import numpy as np

def milestone_3(model, train_dataset, val_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return trainer

# milestone_4.py
from transformers import pipeline

def milestone_4(model, tokenizer, comment):
    # Create a classification pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    # Predict
    predictions = classifier(comment)[0]

    # Convert to probability dictionary
    prob_dict = {cat["label"]: cat["score"] for cat in predictions}
    return prob_dict

# milestone_5.py
import gradio as gr

def milestone_5(model, tokenizer):
    def classify_comment(comment):
        predictions = milestone_4(model, tokenizer, comment)
        return "\n".join([f"{category}: {prob:.2%}" for category, prob in predictions.items()])

    demo = gr.Interface(
        fn=classify_comment,
        inputs=gr.Textbox(lines=2, placeholder="Enter your feedback here..."),
        outputs=gr.Label(num_top_classes=4),
        title="Feedback Classification",
        description="Classify feedback into categories: Pricing, Functionality, Support, UI/UX"
    )
    demo.launch()