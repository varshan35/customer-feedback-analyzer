# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/cymk6sapyj/badge.svg)](https://ask.thili.ai/b/cymk6sapyj)

# ⚖️ Customer Feedback Classifier & Legal Copilot

This workspace houses an NLP pipeline designed to categorize customer feedback into support categories and classify legal drafting clauses.

## 📂 Project Structure

*   **`core_solution.py`**: The core application module containing code for building datasets, tokenization pipelines, DistilBERT sequence classification models, and the Gradio user interface.
*   **`test_core.py`**: A robust unit test suite built using `pytest` to verify dataset splits, Hugging Face configurations, tokenization pipelines, and compilation of the Gradio application UI.

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install torch transformers scikit-learn gradio pytest
```

### 2. Running the Main Solution
To launch the core machine learning classification pipeline and interactive dashboard, run:
```bash
python core_solution.py
```

### 3. Running Unit Tests
Execute the full validation test suite locally on your terminal with:
```bash
python -m pytest test_core.py
```
*(If `pytest` is in your environment's PATH, you can also run `pytest test_core.py` directly)*
## Milestones

Tick these off as you go, and commit after each one.

- [✓] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Start by brainstorming common feedback themes to define your categories and then manually label a small set of example comments.
- [✓] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Refer to the drill's method for extending the tokenizer and model vocabulary with new special tokens for your categories.
- [✓] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Use the Hugging Face Trainer class as shown in the drill to manage the fine-tuning process.
- [✓] **4. Implement an inference function to classify new feedback.**  
      Hint: Adapt the drill's inference pipeline to take new customer comments and generate the appropriate category token.
- [✓] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Use Streamlit to create a basic user interface for inputting text and displaying the model's prediction.

⚖️ Customer Feedback Classifier & Legal Copilot
This workspace houses an NLP pipeline designed to categorize customer feedback into support categories and classify legal drafting clauses.

📂 Project Structure
core_solution.py: The core application module containing code for building datasets, tokenization pipelines, DistilBERT sequence classification models, and the Gradio user interface.
test_core.py: A robust unit test suite built using pytest to verify dataset splits, Hugging Face configurations, tokenization pipelines, and compilation of the Gradio application UI.
🚀 Getting Started
1. Installation
Ensure you have Python installed, then install the required dependencies:

pip install torch transformers scikit-learn gradio pytest
2. Running the Main Solution
To launch the core machine learning classification pipeline and interactive dashboard, run:

python core_solution.py
3. Running Unit Tests
Execute the full validation test suite locally on your terminal with:

python -m pytest test_core.py
(If pytest is in your environment's PATH, you can also run pytest test_core.py directly)
