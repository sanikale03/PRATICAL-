# Credit Card Fraud Detection Summary Report

## 1. Business Problem & Failure of Standard Accuracy
In the domain of digital payments (such as Paytm), credit card fraud detection is a critical business problem. Out of 284,807 transactions in our dataset, only 492 are fraudulent (approximately 0.172%). 

Standard accuracy fails completely in this context. If a model simply predicts all transactions as legitimate, it achieves **99.83% accuracy** while catching 0% of fraud. This would lead to massive financial losses and a complete breakdown of trust. In order to evaluate model performance realistically, we must focus on:
* **Recall (Sensitivity):** The percentage of actual fraud transactions that we successfully flag.
* **Precision:** The percentage of flagged transactions that are actually fraudulent. High precision minimizes false alarms (which annoy customers and waste analyst investigation resources).
* **PR-AUC (Precision-Recall Area Under Curve):** The overall trade-off between Precision and Recall.

---

## 2. Imbalance Handling Strategy
To prevent the models from ignoring the minority class (Fraud), we evaluated three balancing strategies using Logistic Regression:
1. **Original (Unbalanced) Data**
2. **SMOTE (Synthetic Minority Over-sampling Technique):** Oversampling the minority class to 10% of the majority class.
3. **Random Under-sampling (RUS):** Downsampling the majority class to 10 times the minority class.

**SMOTE** was selected as the best imbalance handling strategy because it creates synthetic minority instances rather than simply replicating existing ones. This allows the model to learn a broader decision boundary for fraud patterns, resulting in a significantly higher Recall and F1-score than Logistic Regression on original data, without suffering from the high information loss of undersampling.

---

## 3. Recommended Model & Threshold
We compared three machine learning algorithms: Logistic Regression, Random Forest, and XGBoost.
* **XGBoost Tuned** (hyperparameter-tuned via `RandomizedSearchCV` on PR-AUC) demonstrated the best overall capability with a **PR-AUC of 0.8890**.
* Instead of the default classification threshold of `0.50`, we recommend the **F1-optimal threshold of 0.9558**.

By optimizing the threshold to `0.9558`, we achieved a **Precision of 93.02%** and a **Recall of 81.63%**, which minimizes the number of False Positives (legitimate transactions blocked) while still capturing the vast majority of fraud.

---

## 4. Cost-Benefit Results (per 50,000 Transactions)
Using a confusion matrix simulation on our test set of **56,962 transactions** (standardized below to a standard 50,000 transaction cohort):

| Financial Category | Formulas / Rates | Financial Impact (Test Set - 56,962) | Normalized Impact (Per 50,000) |
| :--- | :--- | :--- | :--- |
| **True Positives (Caught Fraud)** | ₹4,500 saved per caught fraud | ₹360,000 (80 transactions) | **₹316,000** |
| **False Positives (Manual Review)**| ₹150 spent per false flag | ₹900 (6 transactions) | **₹790** |
| **False Negatives (Missed Fraud)** | ₹4,500 lost per missed fraud | ₹81,000 (18 transactions) | **₹71,100** |
| **Investigation Cost (TP + FP)**   | ₹150 spent per flagged review | ₹12,900 (86 transactions) | **₹11,323** |
| **Net Financial Benefit** | Money Saved − Investigation Cost | **₹347,100** | **₹304,677** |

For every **50,000 transactions processed**, deploying the tuned XGBoost model with the `0.9558` threshold creates a net benefit of **₹304,677** compared to having no model at all.

---

## 5. Production Improvements
To transition this model from prototype to Paytm's production ecosystem, we recommend the following enhancements:
1. **Real-time Scoring Service:** Deploy the `fraud_detection_model.pkl` pipeline using a containerized microservice framework (like FastAPI inside Docker) to perform predictions with sub-50ms latency.
2. **Model Drift and Alerting:** Fraud patterns continuously shift. Monitor prediction probability distributions daily. If the Kolmogorov-Smirnov test indicates drift, alert the team.
3. **Feedback Loop and Retraining:** Set up a database loop where analyst review outcomes (e.g., verifying blocked transactions) are fed back into training. Retrain the model monthly on SMOTE-balanced sliding-window datasets to adapt to new fraud patterns.
