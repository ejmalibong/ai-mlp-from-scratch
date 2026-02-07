# MLP From Scratch (NumPy)

## Overview

Multilayer Perceptron (MLP) implemented from scratch using NumPy.  
Pipeline includes:

- data preprocessing
- categorical encoding
- normalization
- train/validation split
- forward/backward propagation
- SGD training loop
- external trainer architecture

Target: regression (student exam score prediction).

---

## Workflow

1. Data pipeline
2. Feature encoding + normalization
3. Train/val split
4. Model initialization
5. Forward pass
6. Loss computation (MSE – Mean Squared Error)
7. Backpropagation
8. Parameter update (SGD – Stochastic Gradient Descent)
9. Evaluation

---

## Folder Structure

```bash
project/
│
├── data/
│ └── student.csv
│
├── notebooks/
│ └── experiments.ipynb
│
├── src/
│ ├── data_pipeline.py
│ ├── model.py
│ └── train.py
│
└── README.md
```

---

## Install

```bash
# Recommended to create a virtual environment
python -m venv venv
source venv/Scripts/ativate

# Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
cd src && python train.py

# OR

jupyter notebook notebooks/mlp_from_scratch_demo.ipynb
```

---

## Design Principles

- model = math only

- training = external loop

- pipeline = separate layer

- no framework dependencies

- no hidden abstractions

- full control of gradients and updates

---
