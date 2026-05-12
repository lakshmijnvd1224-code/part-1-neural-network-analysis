# Part 1 — Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds and analyses a feed-forward neural network to predict
customer churn using structured customer behaviour data.

## Dataset
- **File:** customer_churn_nn.csv
- **Source:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing
- **Rows:** 2000, **Columns:** 17
- **Target:** churn (1 = churned, 0 = retained)

## Approach
1. Loaded and explored the dataset — identified severe class imbalance (1.55% churn rate)
2. Preprocessed data — label encoding, standard scaling, stratified train-test split
3. Built a feed-forward neural network with 2 hidden layers using TensorFlow/Keras
4. Trained with class weights to handle imbalance
5. Ran 3 hyperparameter experiments comparing neurons, layers, learning rate, and activation functions
6. Best result: 97.25% validation accuracy with tanh activation and learning rate 0.01

## Results
| Experiment | Val Accuracy | Val Loss |
|---|---|---|
| Exp 1 Baseline (relu, lr=0.001) | 95.75% | 0.1410 |
| Exp 2 More Layers (relu, lr=0.001) | 96.75% | 0.2229 |
| Exp 3 Higher LR + tanh (lr=0.01) | 97.25% | 0.0987 |

## Repository Structure
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
├── model_comparison_table.png
├── model_comparison_table.csv
└── evaluation_outputs.png

## Libraries Used
- TensorFlow/Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
