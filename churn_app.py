import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Churn Prediction System", page_icon="📊", layout="wide")

# 2. Load Model and Expected Columns
@st.cache_resource
def load_assets():
    # Make sure the path matches your project structure
    model = pickle.load(open('models/xgboost_churn_model.pkl', 'rb'))
    columns = pickle.load(open('models/expected_columns.pkl', 'rb'))
    return model, columns

model, expected_columns = load_assets()

# 3. App UI & Header
st.title("📊 Customer Churn Prediction System")
st.write("Upload a CSV file containing customer data. The AI will analyze it and generate a report of customers at risk of leaving.")

# 4. File Uploader for Batch Prediction
uploaded_file = st.file_uploader("Upload Customer Data (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded data
    raw_data = pd.read_csv(uploaded_file)
    st.write("### 📂 Raw Data Preview")
    st.dataframe(raw_data.head())
    
    if st.button("Predict Churn 🔍"):
        with st.spinner('Analyzing customer data...'):
            # 5. Data Preprocessing Pipeline (Identical to training steps)
            processed_data = raw_data.copy()
            
            # Convert TotalCharges to numeric
            if 'TotalCharges' in processed_data.columns:
                processed_data['TotalCharges'] = pd.to_numeric(processed_data['TotalCharges'], errors='coerce')
            
            # Drop missing values
            processed_data.dropna(inplace=True)
            
            # Save Customer IDs for the final report
            customer_ids = processed_data['customerID'] if 'customerID' in processed_data.columns else processed_data.index
            
            # Drop unnecessary columns
            if 'customerID' in processed_data.columns:
                processed_data.drop(columns=['customerID'], inplace=True)
            if 'Churn' in processed_data.columns: # Just in case the uploaded file has the target column
                processed_data.drop(columns=['Churn'], inplace=True)
                
            # Categorical Encoding
            encoded_data = pd.get_dummies(processed_data, drop_first=True)
            
            # 6. Column Alignment (The Magic Step)
            # Reindex ensures the new data has the EXACT same columns as the training data
            # Missing columns are filled with 0s
            encoded_data = encoded_data.reindex(columns=expected_columns, fill_value=0)
            
            # 7. Make Predictions
            predictions = model.predict(encoded_data)
            
            # 8. Generate and Display Results
            results_df = pd.DataFrame({
                'Customer ID': customer_ids,
                'Prediction': predictions,
                'Risk Status': ['High Risk (Will Leave) 🔴' if p == 1 else 'Low Risk (Will Stay) 🟢' for p in predictions]
            })
            
            st.success("Analysis Complete!")
            st.write("### 🚨 Prediction Results")
            st.dataframe(results_df)
            
            # 9. Download Button for the final report
            csv_report = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report as CSV 📥",
                data=csv_report,
                file_name='churn_risk_report.csv',
                mime='text/csv',
            )