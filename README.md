# 💳 Credit Risk Scoring for Bank GoodCredit

An end-to-end **Machine Learning credit risk scoring system** that predicts whether a customer is likely to be classified as a **bad-risk customer** based on banking account information, enquiry history, demographics, and payment behaviour.

The project combines **SQL data extraction, exploratory data analysis, feature engineering, XGBoost classification, model evaluation, and customer risk ranking** into a single workflow.

---

## 🎯 Objective

The main objective is to build a machine learning model that can:

* Predict high-risk / bad-risk customers
* Use historical account and enquiry behaviour
* Engineer customer-level credit risk features
* Handle class imbalance
* Rank customers according to predicted risk
* Evaluate performance using **ROC-AUC and Gini coefficient**
* Compare model performance against a predefined benchmark

---

## 📊 Data Sources

The project works with three banking datasets:

| Dataset             | Description                                       |
| ------------------- | ------------------------------------------------- |
| `Cust_Account`      | Customer account and payment information          |
| `Cust_Enquiry`      | Customer credit enquiry history                   |
| `Cust_Demographics` | Customer demographic information and target label |

**Target:** `Bad_label`

* `0` → Good-risk customer
* `1` → Bad-risk customer

The notebook also includes a **synthetic fallback dataset** for development/testing when the database is unavailable.

---

## 🔍 Key Features

### Account Features

* Total and average account age
* Total and average account balance
* Total credit limit
* Credit utilisation
* Number of accounts
* Total past-due amount
* Payment-history statistics

### Enquiry Features

* Enquiries in the last 90 days
* Enquiries in the last 365 days
* Total enquiries

### Payment History

Payment-history strings are parsed to derive features such as:

* Average DPD behaviour
* Months until 30+ DPD
* Payment-history length

---

## 🤖 Machine Learning Model

The project uses **XGBoost Classifier** for credit-risk prediction.

The model includes class-imbalance handling using `scale_pos_weight`.

### Model Configuration

* `n_estimators = 300`
* `max_depth = 4`
* `learning_rate = 0.05`
* `subsample = 0.8`
* `colsample_bytree = 0.8`
* Evaluation metric: **AUC**

---

## 📈 Model Evaluation

The model is evaluated using:

* **ROC-AUC**
* **Gini Coefficient**
* **Decile Rank Ordering**

The Gini coefficient is calculated as:

```text
Gini = (2 × AUC - 1) × 100
```

The project uses a **37.9 Gini benchmark** for comparison.

The model also generates a decile ranking where **Decile 10 represents the highest-risk customers**.

---

## 🔄 Project Workflow

```text
Banking Database
       ↓
SQL Data Extraction
       ↓
Data Exploration & Quality Checks
       ↓
Payment History Parsing
       ↓
Customer-Level Feature Engineering
       ↓
Account + Enquiry + Demographic Merge
       ↓
Train/Test Split
       ↓
XGBoost Model
       ↓
Probability Prediction
       ↓
AUC & Gini Evaluation
       ↓
Decile Risk Ranking
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SQLAlchemy
* PyMySQL
* Matplotlib
* Seaborn
* Joblib
* MySQL

---

## 📁 Project Structure

```text
Credit-Risk-Scoring/
│
├── main.ipynb
│
├── data/
│   ├── Cust_Account.csv
│   ├── Cust_Enquiry.csv
│   └── Cust_Demographics.csv
│
├── outputs/
│   ├── model.pkl
│   ├── test_predictions.csv
│   ├── evaluation_report.txt
│   └── decile_rank_order.png
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Credit-Risk-Scoring.git
cd Credit-Risk-Scoring
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost sqlalchemy pymysql matplotlib seaborn joblib
```

### 3. Configure database credentials

Set the required database environment variables:

```text
DB_HOST
DB_PORT
DB_USER
DB_PASSWORD
DB_NAME
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open:

```text
main.ipynb
```

and run the cells sequentially.

---

## 💾 Model Output

The trained XGBoost model is saved as:

```text
outputs/model.pkl
```

Additional outputs include:

```text
outputs/test_predictions.csv
outputs/evaluation_report.txt
outputs/decile_rank_order.png
```

---

## 💼 Business Applications

The model can support:

* Credit underwriting
* High-risk customer identification
* Additional customer verification
* Credit-risk monitoring
* Prioritization of manual reviews
* Data-driven lending decisions

The model is intended as a **decision-support system**, not a replacement for human credit assessment.

---

## ⚠️ Limitations

* Model performance depends on data quality and representativeness.
* Historical customer behaviour may not always predict future behaviour.
* The synthetic dataset is only for development/testing.
* Economic changes may affect model performance.
* Feature importance does not imply causation.
* Additional validation is required before production banking deployment.

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Compare XGBoost with LightGBM and Logistic Regression
* SHAP-based model explainability
* Probability calibration
* Time-based validation
* Model monitoring
* Additional credit-history features
* Interactive credit-risk dashboard
* Automated ML/data pipelines
