# Heart Disease Predictor
Predicts whether a patient has heart disease from medical measurements (such as age, blood pressure and cholesterol).

**File:** [`notebooks/HEART DISEASE DETECTION .ipynb`](../../../notebooks/HEART%20DISEASE%20DETECTION%20.ipynb)

**How it works:** loads `heart_disease_data.csv`, 80/20 stratified train/test split (`random_state=2`), trains scikit-learn Logistic Regression, reports accuracy with `accuracy_score`, and predicts for a new input array of health values.

**Run:** `pip install pandas numpy scikit-learn`, update the CSV path in the notebook, run all cells.
**Not in the repo:** the CSV. No accuracy figure is claimed here.
