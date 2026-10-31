# Sales Prediction Using Python

A machine learning project that analyzes advertising expenditure and predicts product sales using Python. It includes data cleaning, exploratory data analysis, machine learning model comparison, and an interactive Streamlit dashboard.

## Project Overview

The objective of this project is to understand how advertising investments across TV, Radio, and Newspaper influence product sales and to predict sales using machine learning algorithms.

The project uses the Advertising dataset and compares multiple regression models to identify the best-performing model.

## Technologies Used

- **Python** – Programming language
- **Pandas & NumPy** – Data processing and numerical calculations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning and model evaluation
- **Statsmodels** – Statistical analysis
- **Plotly** – Interactive visualizations
- **Streamlit** – Interactive web dashboard
- **Joblib** – Model saving and loading

## Project Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Advertising expenditure and sales visualization
- Correlation analysis
- Training and comparison of machine learning models
- Sales prediction using the trained model
- Interactive Streamlit dashboard
- Business insights based on advertising data

## Machine Learning Models

The following regression algorithms were implemented and evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Model Performance

| Model             |        MAE |       RMSE |   R² Score |
| ----------------- | ---------: | ---------: | ---------: |
| Linear Regression |     1.4608 |     1.7816 |     0.8994 |
| Decision Tree     |     1.3688 |     1.8238 |     0.8946 |
| Random Forest     | **0.6910** | **0.8670** | **0.9762** |

**Best Performing Model:** Random Forest Regressor

The Random Forest model achieved the lowest MAE and RMSE and the highest R² score among the three models on the test split.

## Project Structure

```text
Sales_Prediction/
│
├── data/
│   ├── Advertising.csv
│   └── processed/
│       └── cleaned_sales.csv
│
├── models/
│   └── best_sales_model.pkl
│
├── outputs/
│   ├── charts/
│   └── model_results.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── train_model.py
│   └── predict.py
│
├── dashboard/
│   └── app.py
│
├── inspect_data.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vishal-CSBS/CodeAlpha_SalesPrediction.git
```

### 2. Navigate to the Project Folder

```bash
cd CodeAlpha_SalesPrediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Run the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your web browser.

### Run Individual Python Scripts

**Data Cleaning:**

```bash
python src/data_cleaning.py
```

**Exploratory Data Analysis:**

```bash
python src/eda.py
```

**Train Machine Learning Models:**

```bash
python src/train_model.py
```

**Make Sales Predictions:**

```bash
python src/predict.py
```

## Sample Prediction

Example advertising expenditure:

- TV: 200
- Radio: 40
- Newspaper: 30

**Predicted Sales: 20.04**

*Note: This is an example prediction from the trained model. The result depends on the dataset and model used.*

## Dataset

The project uses the Advertising dataset containing the following features:

| Feature   | Description                          |
| --------- | ------------------------------------ |
| TV        | Advertising expenditure on TV        |
| Radio     | Advertising expenditure on Radio     |
| Newspaper | Advertising expenditure on Newspaper |
| Sales     | Product sales (target variable)      |

## Limitations

- The dataset contains only 200 records.
- Model performance is based on a single train-test split.
- Predictions may vary when applied to new or different datasets.
- The dataset does not contain date information, so time-series forecasting is not included.

## Future Improvements

- Test additional machine learning algorithms.
- Improve model performance using cross-validation and hyperparameter tuning.
- Use larger and more diverse datasets.
- Add more interactive dashboard features.
- Deploy the dashboard online.

## Author

**Vishal.N.A**

- GitHub: (https://github.com/Vishal-CSBS)

* Project: Sales Prediction Using Python
