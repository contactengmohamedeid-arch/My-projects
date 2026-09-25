# Machine Learning & AI Projects Portfolio

This repository contains a collection of end-to-end applied projects in the fields of Machine Learning, Deep Learning, and Natural Language Processing (NLP).

## 🚀 Projects Overview

### 1. Fake News Detection System 📰
*   **Description:** An intelligent system to verify the credibility of news and classify it as genuine or fake[cite: 8].
*   **Technologies & Algorithms:** Text preprocessing was performed to clean links, HTML tags, and punctuation, followed by vectorization using `TfidfVectorizer`[cite: 9]. Several models were trained and compared, including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting, to select the most accurate model[cite: 9].
*   **Interactive UI:** The project features an integrated web interface built with `streamlit` in the `Fake News APP.py` file, allowing users to input news text and select a model to verify its credibility instantly[cite: 8].

### 2. Heart Disease Predictor ❤️
*   **Description:** A predictive model to diagnose heart conditions (healthy or defective) based on medical variables such as age, blood pressure, and cholesterol levels[cite: 10].
*   **Technologies & Algorithms:** The `LogisticRegression` algorithm was used to train the data and build a system that predicts the patient's condition based on an input data array representing their health metrics[cite: 10].

### 3. Cats vs. Dogs Image Classifier (CNN) 🐱🐶
*   **Description:** A deep learning model for image classification to determine whether an image contains a cat or a dog[cite: 11].
*   **Technologies & Algorithms:** A Convolutional Neural Network (CNN) was built using `TensorFlow` and `Keras`[cite: 11]. The network consists of `Conv2D` layers for feature extraction, `MaxPooling2D` for dimensionality reduction, and `Dense` layers for the final classification decision[cite: 11]. `ImageDataGenerator` was utilized to enhance and augment the training data[cite: 11].

### 4. Movie Recommendation Engine 🎬
*   **Description:** A content-based filtering recommendation system designed to suggest 5 movies similar to the one queried by the user[cite: 12].
*   **Technologies & Algorithms:** Movie genres and overviews were combined into a single feature (tags) and converted into vectors using `CountVectorizer`[cite: 12]. The system relies on measuring the `cosine_similarity` between movie vectors to extract and recommend the most closely related movies[cite: 12].

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <Your_GitHub_Repo_URL>
   cd <Repository_Name>