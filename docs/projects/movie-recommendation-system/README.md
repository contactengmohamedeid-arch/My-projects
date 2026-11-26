# Movie Recommendation System
Content-based recommender: give it a movie title, get 5 similar movies.

**File:** [`notebooks/MOVIE RECOMMENDATION SYSTEM.ipynb`](../../../notebooks/MOVIE%20RECOMMENDATION%20SYSTEM.ipynb)

**How it works:** loads `movies.csv`, combines genres and overview into one `tags` text, converts it to vectors with `CountVectorizer` (10,000 features, English stopwords removed), measures cosine similarity between all movies, and `recommend("Iron Man")` returns the closest matches.

**Run:** `pip install pandas scikit-learn`, update the CSV path, run all cells.
**Not in the repo:** `movies.csv`.
