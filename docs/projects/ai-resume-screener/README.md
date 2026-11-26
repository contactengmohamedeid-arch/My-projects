# AI Resume Screener (ATS)
Ranks candidate resumes against a job description using text similarity, with a Streamlit app.

**Files:** [`notebooks/AI Resume Screener (ATS).ipynb`](../../../notebooks/AI%20Resume%20Screener%20%28ATS%29.ipynb), [`ats_app.py`](../../../ats_app.py).

**How it works**
- Loads a resume dataset (`UpdatedResumeDataSet.csv`) and cleans the text (lowercase, removes links, hashtags, mentions, punctuation, non-ASCII characters and English stopwords with NLTK).
- TF-IDF vectors (up to 5,000 features) for the job description and the resumes.
- Cosine similarity gives each resume a score (shown as a percentage), sorted highest first.
- The app has a job-description box and a resume input and shows the match.

**Run:** `pip install pandas scikit-learn nltk streamlit`, then `streamlit run ats_app.py`.
**Not in the repo:** the resume dataset. No accuracy claim is made, this is a similarity ranker, not a trained classifier.
