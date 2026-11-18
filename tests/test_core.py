import pytest
import torch
import gradio as gr
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification, PreTrainedTokenizerBase

# --- Milestone 1: Define 3-5 feedback categories and create a small dataset of example comments with labels.
def test_milestone_1():
    X_train, X_val, y_train, y_val = milestone_1()

    # Test outputs are lists
    assert isinstance(X_train, list)
    assert isinstance(X_val, list)
    assert isinstance(y_train, list)
    assert isinstance(y_val, list)

    # Test dataset sizes are correct (75% train, 25% validation split of 16 comments)
    assert len(X_train) == 12
    assert len(X_val) == 4
    assert len(y_train) == 12
    assert len(y_val) == 4

    # Ensure labels fall within our 4 category indexes (0 to 3)
    assert all(0 <= label <= 3 for label in y_train)

    raise NotImplementedError("write the test for milestone 1")


# Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.

def test_milestone_2():
    X_train, X_val, y_train, y_val = milestone_1()
    tokenizer, train_dataset, val_dataset = milestone_2(X_train, X_val, y_train, y_val)

    # Use the general PreTrainedTokenizerBase to support both fast and slow tokenizers robustly
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert isinstance(train_dataset, FeedbackDataset)
    assert len(train_dataset) == 12

    # Test item loading outputs correct tensor structures
    sample_item = train_dataset[0]
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "labels" in sample_item
    assert isinstance(sample_item["labels"], torch.Tensor)


# Milestone 3: Fine-tune the model using the Hugging Face Trainer.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Implement an inference function to classify new feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Build a simple Streamlit app to input comments and show classifications.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

