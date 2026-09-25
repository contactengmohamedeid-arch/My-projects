import streamlit as st
import pickle
import re

# 1. Page Configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# 2. Text Cleaning Function (Crucial for processing new data)
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\breuters\b', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

# 3. Load Models and Vectorizer (Updated Paths)
@st.cache_resource
def load_vectorizer():
    # إضافة اسم الفولدر models/ قبل اسم الملف
    return pickle.load(open('models/vectorizer.pkl', 'rb'))

@st.cache_resource
def load_models():
    # إضافة اسم الفولدر models/ قبل أسماء كل الموديلات
    models = {
        "Logistic Regression": pickle.load(open('models/LR_model.pkl', 'rb')),
        "Decision Tree (Tuned)": pickle.load(open('models/tuned_DT_model.pkl', 'rb')),
        "Random Forest": pickle.load(open('models/RF_model.pkl', 'rb')),
        "Gradient Boosting": pickle.load(open('models/GB_model.pkl', 'rb'))
    }
    return models

vectorizer = load_vectorizer()
models = load_models()

# 4. App Design and UI
st.title("📰 Fake News Detection System")
st.write("Enter the news article text below to verify whether it is genuine or fake. You can test different machine learning models.")

# Dropdown menu to select the model
selected_model_name = st.selectbox("🤖 Select the Model:", list(models.keys()))
selected_model = models[selected_model_name]

# Text area for the news input
news_input = st.text_area("📝 Enter news text here:", height=200)

# 5. Check Button Logic
if st.button("Check News 🔍"):
    if news_input.strip() == "":
        st.warning("Please enter the news text first!")
    else:
        # Prediction steps: Clean -> Vectorize -> Predict -> Output
        cleaned_news = wordopt(news_input)
        vectorized_news = vectorizer.transform([cleaned_news])
        prediction = selected_model.predict(vectorized_news)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.success(f"✅ Result: This article appears to be **Genuine News**")
        else:
            st.error(f"❌ Result: This article appears to be **Fake News**")
            
        st.info(f"💡 Tested using: {selected_model_name}")