# Customer Churn Prediction
Predicts which telecom customers are likely to leave, with a Streamlit app that scores an uploaded CSV.

**Files:** [`notebooks/Customer Churn Prediction.ipynb`](../../../notebooks/Customer%20Churn%20Prediction.ipynb), [`churn_app.py`](../../../churn_app.py).

**How it works**
- Data: Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). Rows with missing `TotalCharges` are dropped, `customerID` is removed, categorical columns are one-hot encoded.
- 80/20 train/test split, stratified, `random_state=42`.
- Random Forest (100 trees, balanced class weights) and XGBoost (100 estimators, learning rate 0.1, `scale_pos_weight` for class imbalance), each evaluated with a classification report on the test set.
- Top-10 Random Forest feature importances are plotted.
- The XGBoost model and the expected column list are saved with pickle.
- The app takes a CSV of customers, applies the same preprocessing, aligns columns to the training columns and labels each customer High Risk or Low Risk.

**Run:** `pip install pandas scikit-learn xgboost streamlit matplotlib seaborn`, run the notebook to create `models/`, then `streamlit run churn_app.py`.
**Not in the repo:** the dataset and the `models/` folder. Result numbers are not listed here.
