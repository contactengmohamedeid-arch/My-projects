import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure stopwords are available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Page Configuration
st.set_page_config(page_title="AI Resume Screener (ATS)", page_icon="📄", layout="wide")

# 2. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# 3. App UI & Header
st.title("📄 AI Resume Screener & ATS System")
st.write("Evaluate candidate resumes against a specific Job Description using Natural Language Processing (NLP).")

# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Job Description")
    jd_input = st.text_area("Enter the job requirements here:", height=250, placeholder="e.g., Looking for a Python developer with Machine Learning experience...")

with col2:
    st.subheader("📂 Candidate Resume")
    resume_input = st.text_area("Paste the candidate's resume text here:", height=250, placeholder="Paste resume content...")

# 4. Analysis Button Logic
if st.button("Calculate Match Score 🔍"):
    if jd_input.strip() == "" or resume_input.strip() == "":
        st.warning("Please fill in both the Job Description and the Resume text!")
    else:
        with st.spinner("Analyzing and computing similarity..."):
            # Clean both texts
            cleaned_jd = clean_text(jd_input)
            cleaned_resume = clean_text(resume_input)
            
            # Vectorize using TF-IDF
            documents = [cleaned_jd, cleaned_resume]
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate Cosine Similarity between JD (index 0) and Resume (index 1)
            similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            match_percentage = similarity_score * 100
            
            # Display Results
            st.markdown("---")
            st.subheader("🎯 Evaluation Result")
            
            # Metric card display
            st.metric(label="Match Score", value=f"{match_percentage:.2f}%")
            
            if match_percentage >= 40:
                st.success("✅ Strong Match: This candidate's profile aligns well with the job requirements!")
            elif 20 <= match_percentage < 40:
                st.info("⚠️ Moderate Match: The candidate has some relevant skills, but review carefully.")
            else:
                st.error("❌ Weak Match: The candidate's profile may not be suitable for this role.")