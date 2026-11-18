# NLP Tutorial - 15 ML Practicals

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/All_15_Practicals_Colab.ipynb)

Implements all 15 problems in the supplied assignment, in their original order.

- **Submission PDF:** [NLP Tutorial Practical Record](output/pdf/NLP_Tutorial_Practical_Record.pdf).
- **All-in-one notebook:** [All 15 practicals](All_15_Practicals_Colab.ipynb).
- **Individual notebooks:** `notebooks/`; each is standalone and includes embedded text data.
- **Python source:** `src/`; shared preprocessing/evaluation is in `src/common.py`.
- **Measured outputs:** `outputs/` contains charts, text logs, metrics and environment versions.

## Verified cloud execution

All 15 practicals completed in Google Colab on 18 November 2026. The downloaded cloud notebook contains 15 output charts, no error outputs, and the final completion marker. Colab omits execution counts from its downloaded file; counts 1 through 47 were observed in the live UI.

- [Executed cloud notebook](All_15_Practicals_Colab_Executed.ipynb)
- [Personal Drive copy](https://colab.research.google.com/drive/1C1L7ZWiY5o6PdAdhhsLEtXzLOLQRF26Q)

The PDF reports the reproducible local run. Cloud outputs are preserved separately; small version-dependent differences are expected.

## Run in Google Colab

1. Click the Colab badge above (a standard CPU runtime is sufficient).
2. Choose **Runtime > Run all**. Approve execution of this notebook if Colab asks.
3. The notebook creates its teaching datasets automatically and prints every result.
4. Use **File > Save a copy in Drive** if you want a personal editable copy.

## Practical index

| No. | Practical | Required model | Notebook |
|---|---|---|---|
| 1 | House Price Prediction | Linear Regression | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/01_house_price_prediction.ipynb) |
| 2 | Loan Approval Prediction | Logistic Regression | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/02_loan_approval_prediction.ipynb) |
| 3 | Iris Flower Classification | K-Nearest Neighbors (KNN) | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/03_iris_flower_classification.ipynb) |
| 4 | Employee Promotion Prediction | Decision Tree | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/04_employee_promotion_prediction.ipynb) |
| 5 | Customer Churn Prediction | Random Forest | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/05_customer_churn_prediction.ipynb) |
| 6 | Breast Cancer Classification | Support Vector Machine (SVM) | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/06_breast_cancer_classification.ipynb) |
| 7 | Spam SMS Detection | TF-IDF + Naive Bayes | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/07_spam_sms_detection.ipynb) |
| 8 | Customer Segmentation | K-Means Clustering | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/08_customer_segmentation.ipynb) |
| 9 | Used Car Price Prediction | Random Forest Regressor | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/09_used_car_price_prediction.ipynb) |
| 10 | Credit Card Fraud Detection | Isolation Forest | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/10_credit_card_fraud_detection.ipynb) |
| 11 | Student Performance Prediction | Gradient Boosting Regressor | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/11_student_performance_prediction.ipynb) |
| 12 | Movie Review Sentiment Analysis | CountVectorizer + SVM | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/12_movie_review_sentiment_analysis.ipynb) |
| 13 | Dimensionality Reduction of Handwritten Digits | Principal Component Analysis (PCA) | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/13_dimensionality_reduction_of_handwritten_digits.ipynb) |
| 14 | News Category Prediction | TF-IDF + Multinomial Logistic Regression | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/14_news_category_prediction.ipynb) |
| 15 | Sales Prediction | AdaBoost Regressor | [Open in Colab](https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/notebooks/15_sales_prediction.ipynb) |

## Data and evaluation

The assignment specifies models and features but supplies no datasets. Iris, breast cancer and digits use scikit-learn's packaged benchmark data. Other numerical data are generated from documented, seeded rules. SMS, movie reviews and news use original, manually authored teaching corpora (80, 80 and 100 rows respectively). These are **demonstrations, not production benchmarks**. No personal SMS, private records or scraped reviews are included.

Supervised tasks use a 75/25 split, stratified for classification. Preprocessing is fitted on training data. Sales uses the first 75 percent for training and evaluates rolling one-day-ahead predictions on the remaining dates. Fraud labels are withheld from Isolation Forest fitting. Clustering uses internal silhouette evaluation; PCA reports retained variance and held-out reconstruction error. All random seeds are 42. Hyperparameters are fixed rather than selected using test labels.

## Run locally and rebuild

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build.py
python make_report.py
```

Run a single Python practical from the repository root, for example:

```bash
python src/03_iris_flower_classification.py
```

The saved notebook outputs are captured from actual code execution. `outputs/environment.json` records tested versions. `requirements.txt` provides compatible version ranges for Colab; `requirements-tested.txt` records exact local scientific package versions. Results can vary slightly across versions/platforms.

## References

- [scikit-learn dataset documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html)
- [scikit-learn model and preprocessing guide](https://scikit-learn.org/stable/user_guide.html)
- [Google Colab](https://colab.research.google.com/)

Original assignment: *15 ML Tutorial Problem Statements.pdf*, supplied by the student. The source PDF is not redistributed.
