# Part 1 — Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds and analyses a feed-forward neural network to predict
customer churn using structured customer behaviour data.

## Dataset
- **File:** customer_churn_nn.csv
- **Source:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing
- **Rows:** 2000, **Columns:** 17
- **Target:** churn (1 = churned, 0 = retained)
- **Note:** Dataset file is not uploaded to this repository. Please download from the source link above.

## Approach
1. Loaded and explored the dataset — identified severe class imbalance (1.55% churn rate)
2. Preprocessed data — label encoding, standard scaling, stratified train-test split
3. Built a feed-forward neural network with 2 hidden layers using TensorFlow/Keras
4. Trained with class weights to handle class imbalance
5. Ran 3 hyperparameter experiments comparing neurons, layers, learning rate, and activation functions
6. Saved evaluation outputs and comparison table in results folder

## Results
| Experiment | Val Accuracy | Val Loss |
|---|---|---|
| Exp 1 Baseline (relu, lr=0.001) | 95.75% | 0.1410 |
| Exp 2 More Layers (relu, lr=0.001) | 96.75% | 0.2229 |
| Exp 3 Higher LR + tanh (lr=0.01) | 97.25% | 0.0987 |

## Observations
- The dataset is severely imbalanced with only 1.55% churn rate (31 out of 2000 customers)
- Class weights were applied during training to prevent the model from ignoring the minority class
- The model achieved 97% overall accuracy but struggled to detect churned customers due to very few examples
- Training accuracy reached 99% while validation accuracy stabilised at 96-97%, indicating mild overfitting
- Experiment 3 with tanh activation and higher learning rate of 0.01 gave the best result
- ReLU activation worked well but tanh performed slightly better on this dataset
- A higher learning rate of 0.01 converged faster than 0.001 without causing instability

## Repository Structure
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.png or .csv
    └── evaluation_outputs.png

## Libraries Used
tensorflow — for building the neural network

scikit-learn — for preprocessing, train-test split, metrics

pandas — for loading and handling data

numpy — for numerical operations

matplotlib — for plotting graphs

seaborn — for visualisations like heatmap
