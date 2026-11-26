# Fake News Detection
Classifies news articles as real or fake, with a Streamlit app where you paste text and pick a model.

**Files:** [`notebooks/FAKE NEWS DETECTION.ipynb`](../../../notebooks/FAKE%20NEWS%20DETECTION.ipynb) (training), [`notebooks/Fake News APP.py`](../../../notebooks/Fake%20News%20APP.py) (app).

**How it works**
- Loads two CSV files, `True.csv` and `Fake.csv`, and combines them.
- Cleans text with a `wordopt` function (lowercase, removes links, HTML tags, punctuation, digits, the word "reuters").
- TF-IDF vectorization, 70/30 train/test split.
- Trains and compares Logistic Regression, Decision Tree, Random Forest and Gradient Boosting; models are saved with pickle.
- The app loads the saved vectorizer and four models from a `models/` folder and lets you choose one.

**Run:** `pip install pandas scikit-learn streamlit`, run the notebook to create the `models/` files, then `streamlit run "notebooks/Fake News APP.py"`.
**Not in the repo:** the CSV data and the `models/` folder.
