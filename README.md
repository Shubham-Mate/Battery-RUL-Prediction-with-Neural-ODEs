<p align="center"><h1 align="center">Battery RUL Prediction with Neural ODE based GRU</h1></p>
<p align="center">
	<img src="https://img.shields.io/github/license/Shubham-Mate/Battery-RUL-Prediction-with-Neural-ODEs?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Shubham-Mate/Battery-RUL-Prediction-with-Neural-ODEs?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Shubham-Mate/Battery-RUL-Prediction-with-Neural-ODEs?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Shubham-Mate/Battery-RUL-Prediction-with-Neural-ODEs?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

---

# 📍 Overview

In this project, we developed a predictive model for estimating the Remaining Useful Life (RUL) of batteries using a combination of Neural ODE and Gated Recurrent Units (GRU). Accurate battery RUL predictions are essential for optimizing battery usage, ensuring safety, and minimizing maintenance costs, especially in critical applications such as electric vehicles and energy storage systems.
  Problem Statement:
    Predicting battery RUL involves estimating the time remaining before the battery's performance degrades below an acceptable threshold.

### 1. Proposed Solution:
The integration of Neural ODE into GRU-based architectures where the Neural ODE interpolates the hidden state between two time inputs. This approach leverages the strengths of ODEs in modeling continuous-time dynamics and GRUs in capturing temporal dependencies in time-series data. The hybrid model is designed to predict the complex and nonlinear degradation behavior of batteries.

### 2. Model Architecture:
 - GRU: Utilized to process sequential battery data and extract meaningful temporal features.
 - Neural ODE Component: Introduced to explicitly incorporate battery degradation dynamics by embedding differential equations into the GRU’s hidden states.

### 3. Dataset and Preprocessing:
Data was obtained from [this Kaggle Dataset](https://www.kaggle.com/datasets/ignaciovinuales/battery-remaining-useful-life-rul).
Preprocessing steps, such as feature creation, normalization, and smoothing, were performed to ensure model robustness.

### 4. Training and Evaluation:
The model was trained using a combination of supervised learning techniques and loss functions tailored for time-series regression.
Evaluation metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R2 Score were used to assess performance.

### Conclusion:

The ODE-based GRU framework represents a step forward in battery health management systems, offering scalable and accurate RUL predictions for modern energy solutions.

---


# 📁 Project Structure

```sh
└── Battery-RUL-Prediction-with-Neural-ODEs/
    ├── Neural ODE Experiments.ipynb
    ├── constants.py
    ├── data
    │   └── Final Database.csv
    ├── model.py
    ├── report
    │   ├── Report.aux
    │   ├── Report.log
    │   ├── Report.pdf
    │   ├── Report.synctex.gz
    │   ├── Report.tex
    │   └── imgs
    ├── requirements.txt
    └── util_functions.py
```

---
# 🙌 References

The papers which were reviewed for this project were:
- [Neural Ordinary Differential Equations (Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud)](https://arxiv.org/abs/1806.07366)
- [Neural Ordinary Differential Equation based Recurrent Neural Network Model (Mansura Habiba, Barak A. Pearlmutter)](https://arxiv.org/abs/2005.09807)

---
