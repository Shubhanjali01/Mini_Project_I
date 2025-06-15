## Project Title:
Traffic Forecasting using Spatio‑Temporal Data

## 🧠 Objective:
To design a machine learning model that accurately predicts future traffic conditions (e.g., speed) by learning complex spatial and temporal dependencies in road network data.

## 📚 Reference Research:

![Spatio-Temporal Graph Neural Network for Traffic Forecasting (ASTGCN)](https://arxiv.org/abs/2108.09091)

![Evaluation of Deep Learning for Short-Term Traffic Forecasting)](https://www.mdpi.com/1424-8220/24/20/6659)

## 📦 Dataset Used:
METR-LA (Los Angeles Traffic Dataset)

Source: Loop sensors in Los Angeles highways

Features: Traffic speed readings across multiple sensors over time

## 🧰 Technologies Used:
Languages: Python

Libraries: PyTorch / TensorFlow, DGL / PyTorch Geometric, Pandas, NumPy, Scikit-learn

Models: GCN, GRU, LSTM, ASTGCN (Attention-based Spatio-Temporal Graph Convolutional Network)


## 🔧 Methodology:
`Data Preprocessing`

1. Handle missing values and normalize sensor readings
2. Create adjacency matrix based on road distances

`Model Implementation`

1. Use GCN (Graph Convolution Networks) to model spatial correlations
2. Use GRU/LSTM for temporal dynamics
3. Integrate Attention mechanisms to weigh the importance of different time steps and locations

`Training & Evaluation`

1. Train on past traffic patterns to predict future speeds
2. Metrics used: MAE, RMSE, MAPE

`Suggested Enhancements (optional if implementing for extra credit):`

1. Add weather/incident data for multivariate modeling
2. Experiment with cross-city generalization using PeMS datasets
3. Try temporal attention + spatial attention separately

###  Results:
Achieved MAE: X.XX, RMSE: X.XX on the METR-LA dataset

Attention-based models outperformed baseline GCN + GRU combinations
