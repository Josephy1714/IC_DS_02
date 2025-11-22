# IC_DS_02 – Credit Card Fraud Detection

## Overview
This project is part of the Interncred Data Science Internship.  
We build a fraud detection system using the **Credit Card Fraud dataset** (~285k transactions, 65MB).  
The dataset is highly imbalanced (492 fraud vs 284,315 non‑fraud).

👉 You can view the full notebook on [Google Colab](https://colab.research.google.com/drive/1hIK8Y16a_SgbeHLti8SAwHSEAuNF_cNa?usp=sharing).

## Workflow
1. **Data Loading & Inspection** – confirmed 284,807 rows, 31 columns, no missing values.
2. **Preprocessing** – scaled `Amount` and `Time`, kept PCA features `V1–V28`.
3. **Train/Test Split** – 80/20 split with stratification.
4. **Imbalance Handling** – applied **SMOTE** to balance fraud vs non‑fraud in training set.
5. **Models Tested**:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - Isolation Forest (unsupervised)
   - Autoencoder (unsupervised)

## Results

| Model              | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC‑AUC |
|--------------------|-------------------|----------------|------------|---------|
| Logistic Regression | 0.06              | **0.92**       | 0.11       | 0.97    |
| Decision Tree       | 0.34              | 0.78           | 0.47       | 0.89    |
| Random Forest       | **0.81**          | 0.81           | **0.81**   | 0.97    |
| XGBoost             | 0.73              | **0.89**       | 0.80       | **0.98** |
| Isolation Forest    | 0.31              | 0.33           | 0.32       | 0.66    |
| Autoencoder         | 0.14              | 0.85           | 0.24       | 0.92    |

## Conclusion
- **XGBoost** achieved the best performance (Recall = 0.89, ROC‑AUC = 0.98).
- **Random Forest** also performed strongly with balanced precision and recall.
- **Logistic Regression** had very high recall but unusable precision.
- **Decision Tree** was moderate, while anomaly detection methods (Isolation Forest, Autoencoder) were weaker.
- Final submission file **submission1.csv** was generated using XGBoost predictions.

## Author
- [Joseph Ombati](https://www.linkedin.com/in/joseph-ombati-623aa8381)  
- GitHub: [Josephy1714](https://github.com/Josephy1714)
