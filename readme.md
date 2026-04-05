# Mini Tokenizer Training Demo

A minimal end-to-end pipeline demonstrating how to train a tokenizer from scratch.

This project is intended for **educational purposes** and demonstrates the core components of modern LLM systems.

Pipeline:

Corpus → Tokenizer Training

---

# Features

- Train a **BPE tokenizer** from raw text
- Minimal and easy to understand code

---


# Environment Setup

This project uses **Python venv**.

## 1 Create virtual environment

~~~
python3 -m venv venv
~~~

## 2 Activate environment

Mac / Linux

~~~
source venv/bin/activate
~~~

Windows

~~~
venv\Scripts\activate
~~~

## 3 Install dependencies

~~~
pip install -r requirements.txt
~~~

---

# Train Tokenizer

Train tokenizer from raw corpus:

Download data on https://huggingface.co/datasets/Skylion007/openwebtext/tree/main/plain_text

~~~
python tokenizer/train_tokenizer.py
~~~

Output:

~~~
tokenizer.json
~~~

Tokenizer type:

- Byte-level BPE  
- Vocabulary size: 16000

---

# Train Model

Train a small transformer language model:

the basic training scipt is uploaded as train.py, yet the data needs to be preprocessed into tokenized .pt format.
I trained it on autoDL's NV A800 GPU, and the model is uploaded to https://huggingface.co/Hippocrene/MiniLLM-0.1B

---


# Notes

This project is a **minimal demonstration** and not intended for production use.

Real LLM training typically requires:

- billions of tokens
- larger transformer models
- distributed GPU training

---

# License

MIT License
