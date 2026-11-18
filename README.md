# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/cymk6sapyj/badge.svg)](https://ask.thili.ai/b/cymk6sapyj)

This Customer Feedback Analyzer helps retail managers quickly identify trends and prioritize improvements by automatically categorizing customer feedback comments. It uses a fine-tuned GPT-2 model to classify comments into predefined categories like 'product issue' or 'staff praise', demonstrating an end-to-end inference pipeline.

## What you are building

**Goal.** You will build a tool that automatically categorizes customer feedback comments using a fine-tuned GPT-2 model, proving your ability to apply fine-tuning logic and an inference pipeline to classify new customer comments.

**Done when.** The fine-tuned GPT-2 model can classify new customer comments by generating special label tokens. A simple Streamlit app allows users to input comments and view their classifications.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Start by brainstorming common feedback themes to define your categories and then manually label a small set of example comments.
- [ ] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Refer to the drill's method for extending the tokenizer and model vocabulary with new special tokens for your categories.
- [ ] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Use the Hugging Face Trainer class as shown in the drill to manage the fine-tuning process.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Adapt the drill's inference pipeline to take new customer comments and generate the appropriate category token.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Use Streamlit to create a basic user interface for inputting text and displaying the model's prediction.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Add more feedback categories and expand the dataset with more diverse examples.
- Implement a feature to display a summary dashboard of feedback trends (e.g., top categories, sentiment over time).
- Allow users to edit or correct classifications within the Streamlit app to improve future model performance.
