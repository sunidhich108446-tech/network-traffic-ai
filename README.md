# 📡 Network Intelligence Dashboard
### ML-Powered Network Latency Forecasting System

An interactive Machine Learning application that predicts network latency (Round-Trip Time / RTT) **30 seconds into the future** using XGBoost Regression.

The project explores how historical network behaviour and performance metrics can be used to forecast upcoming latency. It combines machine learning experimentation, feature engineering, model evaluation and an interactive Streamlit dashboard to make network latency forecasting easier to understand and explore.

## Overview

Network latency is an important factor affecting the performance and reliability of computer networks.

Instead of only observing current network conditions, this project explores a predictive approach to estimate future network latency.

The system uses historical network measurements and engineered features to predict the **RTT after 30 seconds**.

The final forecasting model uses **XGBoost Regression**, selected after experimenting with multiple machine learning algorithms.

The trained model is integrated into an interactive Streamlit dashboard where users can enter network measurements, generate predictions, simulate hypothetical network conditions and explore historical latency trends.

##Problem Statement

Traditional network performance analysis often focuses on monitoring existing network conditions.

However, anticipating possible latency changes can provide additional insight into network behaviour.

The objective of this project is to:

- Analyse historical network performance measurements.
- Identify patterns and relationships between network metrics.
- Predict future RTT using machine learning.
- Explore how changes in network conditions influence predicted latency.
- Present predictions through an interactive dashboard.

## Key Features

### 1. 30-Second Latency Forecasting
- Predicts future Round-Trip Time (RTT) using an XGBoost regression model.
- Uses network performance indicators as input features.
- Provides predictions in milliseconds.

### 2. Machine Learning Model Experimentation
- Experimented with multiple regression algorithms.
- Evaluated candidate models using regression performance metrics.
- Selected XGBoost Regression as the final forecasting model.

### 3. Interactive Network Dashboard
- Accepts network measurements through interactive input fields.
- Displays predicted RTT alongside current network parameters.
- Provides an easy-to-understand overview of network conditions.

### 4. What-if Scenario Simulator
- Allows users to modify selected network parameters.
- Generates a new prediction for the hypothetical scenario.
- Compares the original prediction with the modified prediction.
- Visualizes the difference between both forecasts.

### 5. Historical Latency Visualization
- Displays historical RTT measurements.
- Helps explore latency trends over time.
- Uses timestamp-based historical data.

### 6. Forecast Report Export
- Generates a downloadable CSV report.
- Includes original network inputs, forecast results and simulated scenario values.
- Makes it easier to retain and analyse prediction results.

---

##  Machine Learning Approach

The project follows a structured machine learning workflow.

### Step 1: Data Preparation

- Loaded and inspected the network traffic dataset.
- Handled missing values and prepared numerical features.
- Prepared the target variable for supervised regression.

### Step 2: Feature Engineering

Created features to capture recent network behaviour, including:

- Current average RTT.
- Previous RTT measurements.
- Network jitter.
- Packet loss percentage.
- Rolling latency statistics.
- Latency variation.

### Step 3: Model Experimentation

Multiple machine learning regression algorithms were explored and evaluated.

The objective was to identify a suitable model for predicting future network latency.

### Step 4: Final Model Selection

**XGBoost Regressor** was selected as the final forecasting model.

XGBoost is a gradient-boosting algorithm that builds an ensemble of decision trees sequentially, with each new tree improving upon the errors of the existing ensemble.

It is particularly useful for modelling complex relationships in structured tabular data.

### Step 5: Model Evaluation

The model development process uses regression evaluation metrics such as:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error. |
| RMSE | Penalizes larger prediction errors more heavily. |
| R² Score | Measures the proportion of target variance explained by the model. |

A chronological 80:20 train-test split is used for the time-dependent forecasting task.

*Note: Actual evaluation scores should be reported from the final model evaluation rather than assumed.*

---

##  Dataset and Features

The project uses a network performance dataset containing approximately **49,532 records** after preparing the target data.

The target variable is:

`target_rtt_30s`

This represents the RTT value 30 seconds ahead.

### Input Features

| Feature | Description |
|---|---|
| `rtt_avg_ms` | Current average RTT. |
| `rtt_avg_30s_ago` | Average RTT recorded 30 seconds earlier. |
| `rtt_avg_60s_ago` | Average RTT recorded 60 seconds earlier. |
| `jitter_ms` | Variation in network packet delay. |
| `packet_loss_pct` | Percentage of packets lost. |
| `latency_mean_5` | Rolling mean of latency. |
| `latency_max_5` | Maximum latency within the rolling window. |
| `latency_std_5` | Rolling standard deviation of latency. |
| `latency_change` | Change in latency. |

The dataset is used for supervised learning, where historical network measurements are mapped to future RTT values.

---

## Application Workflow

```text
Historical Network Dataset
          |
          v
Data Cleaning & Preprocessing
          |
          v
Feature Engineering
          |
          v
Train-Test Split
(Chronological 80:20)
          |
          v
Experiment with ML Algorithms
          |
          v
XGBoost Regression
          |
          v
Model Evaluation
          |
          v
Save Trained Model
          |
          v
Streamlit Dashboard
          |
          v
30-Second RTT Forecast
          |
          v
What-if Simulation & CSV Export
```

---

##  Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Model evaluation and ML utilities |
| XGBoost |Random Forest| AdaBoost| Gradient Boosting |Final regression model |
| Matplotlib / Seaborn | Exploratory data analysis and visualization |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |
| Git & GitHub | Version control and project hosting |

---

##  Project Structure

```text
network-traffic-ai/
│
├── app.py
│
├── src/
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── models/
│   └── latency_model.joblib
│
├── data/
│   └── raw/
│       └── final.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Dataset and trained model files may need to be obtained separately if they are excluded from the Git repository.

---

##  Limitations

- The application currently accepts manually entered network measurements rather than collecting live network traffic.
- Historical visualizations are based on the available dataset, not continuous live monitoring.
- Predictions are estimates generated by a trained machine learning model and are not guaranteed network measurements.
- Model performance depends on the quality and representativeness of the training data.

---

## Future Improvements

Some possible extensions of this project include:

- Integration with live network monitoring tools.
- Continuous network data collection and automated forecasting.
- Experimentation with advanced time-series forecasting approaches.
- Real-time anomaly detection for unusual latency patterns.
- Model explainability using feature importance and SHAP.
- Improved model monitoring and automated retraining.

---

## Author

**Sunidhi Chaurasia**

Electronics and Communication Engineering Student

- GitHub: [@sunidhich108446-tech](https://github.com/sunidhich108446-tech)

---

⭐ If you find this project interesting, consider giving the repository a star!
