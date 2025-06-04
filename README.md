## Project Overview
This repository contains all code, data, and documentation for **Beedie Analytics Hackathon 2025**—an interdisciplinary analytics challenge hosted by SFU’s Beedie School of Business. Our objective was to develop a data-driven solution addressing the business problem defined by the organizers.

---

## Background & Motivation
- **Problem Statement**  
  Describe the core challenge posed during this hackathon (e.g., “Optimize retail inventory allocation using predictive analytics”).
- **Why It Matters**  
  Explain why solving this problem has tangible benefits—reducing stockouts, improving customer satisfaction, or minimizing holding costs.
- **Approach Summary**  
  Provide a high-level overview of your solution—whether it’s a machine learning model, interactive dashboard, or optimization algorithm.

---

## Features
1. **Data Ingestion & Cleaning**  
   Scripts to load raw data, handle missing values, and normalize features for downstream modeling.
2. **Exploratory Data Analysis (EDA)**  
   Jupyter notebooks showcasing key visualizations, statistical summaries, and correlation analyses.
3. **Model Training & Evaluation**  
   Implementation of one or more predictive models (e.g., Random Forest, XGBoost) with hyperparameter tuning and performance metrics.
4. **Interactive Dashboard (Optional)**  
   A web-based dashboard (built with Streamlit or Dash) allowing users to explore results and scenario-test parameters.
5. **Deployment Scripts**  
   Instructions/API endpoints to deploy the final model as a RESTful service or containerized application.

---

## Installation
These steps assume you have Python 3.7+ installed.

1. **Clone the repository**  
   ```bash
   git clone https://github.com/siddhantj12/Beedie-Hackathon.git
   cd Beedie-Hackathon
````

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Docker Setup**

   ```bash
   docker build -t beedie-hackathon .
   docker run -p 8501:8501 beedie-hackathon
   ```

---

## Usage

1. **Data Preprocessing**

   ```bash
   python scripts/preprocess_data.py \
       --input data/raw/your_data.csv \
       --output data/processed/clean_data.csv
   ```
2. **Run Exploratory Data Analysis**

   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```
3. **Train Model**

   ```bash
   python scripts/train_model.py \
       --train_data data/processed/clean_data.csv \
       --model_output models/final_model.pkl
   ```
4. **Evaluate Model**

   ```bash
   python scripts/evaluate_model.py \
       --model models/final_model.pkl \
       --test_data data/processed/test.csv
   ```
5. **Launch Dashboard (if available)**

   ```bash
   streamlit run app/dashboard.py
   ```

---

## Data

* **Raw Data Source**

  * *Filename:* `data/raw/your_data.csv`
  * *Description:* Briefly summarize the dataset (e.g., “Sales transactions for a major retailer from Jan 2020 to Dec 2021”).
* **Processed Data**

  * All intermediate features, cleaned tables, and train/test splits are stored under `data/processed/`.
* **Data Dictionary**

  * File: `data/data_dictionary.md`
  * Explains each column, data type, and possible values.

---

## Modeling & Implementation

1. **Algorithms Explored**

   * *Random Forest:* Baseline tree-based model for feature importance and interpretability.
   * *XGBoost:* Gradient boosting for improved predictive performance.
   * *Linear Models:* Lasso/Ridge for baseline regression/classification.
2. **Evaluation Metrics**

   * *Regression:* RMSE, MAE, $R^2$.
   * *Classification:* Accuracy, Precision, Recall, AUC-ROC.
3. **Hyperparameter Tuning**

   * Used scikit-learn’s `GridSearchCV` with 5-fold cross-validation.
4. **Final Model Performance**

   * Summarize best-in-class results (e.g., “Our XGBoost model achieved an RMSE of 12.5 on the holdout set”).

---

## Project Structure

```
Beedie-Hackathon/
├── data/
│   ├── raw/
│   │   └── your_data.csv
│   ├── processed/
│   │   ├── clean_data.csv
│   │   └── test.csv
│   └── data_dictionary.md
├── models/
│   └── final_model.pkl
├── notebooks/
│   └── EDA.ipynb
├── scripts/
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── evaluate_model.py
├── app/
│   └── dashboard.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Team Members

* **Siddhant Jain**
* **Enya Zeng**
* **Ryan Lee**

---

## How to Contribute

1. **Fork the Repository** and create a feature branch:

   ```bash
   git checkout -b feature/YourFeatureName
   ```
2. **Make Your Changes** (code, notebooks, or documentation).
3. **Ensure Tests Pass** (if any tests exist).
4. **Submit a Pull Request** with a clear description of your contribution.
5. **Maintain Code Style:** Follow PEP 8 for Python, add docstrings, and use consistent notebook formatting.

---

## Acknowledgments

* **SFU Beedie School of Business** for hosting the hackathon.
* **Open-Source Libraries:** scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Streamlit.
* **README Inspiration:** Derived structure from common hackathon README best practices.


<p align="center">“Made with ♥ at SFU Beedie Hackathon 2025”</p>
