# 🎓 Student Academic Assistant: A Conversational AI Approach Using GPT-2 & BERT

A conversational AI project comparing a generative model (GPT-2) and a classification model (BERT) for building a chatbot that answers university course-related questions. Fine-tuned on a custom question-answer dataset built from the University of Illinois Course Catalog.

## Overview
The goal is to build an academic assistant capable of answering student queries about courses — names, descriptions, credit hours, and subject categories — and to compare how a generative transformer (GPT-2) stacks up against a classification-based transformer (BERT) for this task.

The project covers:

* **Dataset Construction** — building a custom QA dataset (23,000 pairs) from the course catalog
* **Preprocessing** — handling missing values, text formatting, tokenization/truncation (max 128 tokens), label encoding for BERT
* **Model Fine-Tuning** — GPT-2 (`GPT2LMHeadModel`) as a generative model, BERT (`BertForSequenceClassification`) as a classifier
* **Evaluation** — training loss/accuracy tracking, and manual real-world testing with user corrections
* **Visualization** — training accuracy and loss curves for both models

## Dataset
`course-catalog.csv` — University of Illinois Course Catalog, used to derive 23,000 QA pairs covering:

* Course names (e.g., "What is the name of the course CS 101?")
* Course descriptions
* Credit hours
* Subject categories

Source: https://discovery.cs.illinois.edu/dataset/course-catalog/

## Results

| Metric | GPT-2 | BERT |
|---|---|---|
| Final Accuracy | **72.61%** | 38.61% |
| Training Loss Reduction | Significant | Moderate |
| Performance on Structured Queries | Moderate | Excellent |
| Performance on Open-ended Questions | Excellent | Poor |
| Adaptability to New Queries | High | Limited |
| Manual Correction Requirement | Less frequent | More frequent |

* **GPT-2** generated more natural, flexible responses and adapted well to rephrased or unseen questions, though it occasionally produced verbose or slightly off-topic answers.
* **BERT** was highly accurate on fixed, structured questions (e.g., credit hours, subject codes) but struggled with varied phrasing and required more manual correction.
* Overall, GPT-2 was better suited for open-ended conversational use, while BERT worked well as a narrow, fact-lookup component.

## Tech Stack

* **Language**: Python (Google Colab, GPU runtime)
* **Libraries**: `transformers` (GPT-2, BERT), `torch`, `pandas`, `matplotlib`

## Techniques Implemented

* Custom QA dataset generation from structured course data
* Text preprocessing: missing value handling, formatting, tokenization, truncation (128 tokens)
* Label encoding for BERT classification (`label_to_answer.json` mapping)
* Fine-tuning `GPT2LMHeadModel` with causal language modeling (AdamW, lr=5e-5, 3 epochs)
* Fine-tuning `BertForSequenceClassification` (AdamW, lr=3e-5, 2 epochs)
* Training accuracy/loss visualization
* Manual interactive testing loop with answer correction and feedback

## Project Structure

```
student-academic-assistant/
├── notebooks/
│   └── student_assistant_gpt2_bert.ipynb   # Full training & evaluation notebook (Python)
└── data/
    └── course-catalog.csv                   # Source course catalog dataset
```

## How to Use

1. Open the notebook in Google Colab (recommended, GPU runtime) or Jupyter:

```
jupyter notebook notebooks/student_assistant_gpt2_bert.ipynb
```

2. Make sure `course-catalog.csv` is available in the working directory (update the file path if needed, e.g. `data/course-catalog.csv`).

3. Run the cells in order:
   - Build the custom QA dataset from the catalog
   - Fine-tune BERT for classification
   - Fine-tune GPT-2 for generation
   - Visualize training accuracy/loss
   - Test either model interactively with your own questions

## Author
Rawan Mansour

This was a team project.
