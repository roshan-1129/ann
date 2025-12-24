import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# ================= CUSTOM STYLES =================
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    text-align: center;
}
.metric-title {
    font-size: 16px;
    color: #7f8c8d;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #2c3e50;
}
.risk-high {
    color: #e74c3c;
    font-weight: bold;
}
.risk-low {
    color: #27ae60;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("ann_churn_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

# ================= HEADER =================
st.title("📊 Customer Churn Analytics Dashboard")
st.write(
    "Predict customer churn using an **Artificial Neural Network (ANN)**. "
    "This dashboard simulates how ML models are used in real-world business decision systems."
)

st.divider()

# ================= SIDEBAR =================
st.sidebar.header("🧾 Customer Profile")

credit_score = st.sidebar.number_input("Credit Score", 300, 850, 600)
geography = st.sidebar.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.sidebar.selectbox("Gender", label_encoder.classes_)
age = st.sidebar.slider("Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input("Account Balance", min_value=0.0, step=500.0)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.radio("Has Credit Card", ["Yes", "No"])
is_active_member = st.sidebar.radio("Active Member", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=1000.0)

# ================= DATA PREPARATION =================
gender_encoded = label_encoder.transform([gender])[0]

input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active_member == "Yes" else 0],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder.get_feature_names_out(["Geography"])
)

input_df = pd.concat([input_df, geo_df], axis=1)
input_df = input_df[scaler.feature_names_in_]
input_scaled = scaler.transform(input_df.values)

# ================= PREDICTION =================
prediction = model.predict(input_scaled)
churn_prob = float(prediction[0][0])
threshold = 0.30

# ================= KPI SECTION =================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Churn Probability</div>
            <div class="metric-value">{churn_prob*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    risk_level = "High Risk" if churn_prob > threshold else "Low Risk"
    risk_class = "risk-high" if churn_prob > threshold else "risk-low"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Risk Level</div>
            <div class="metric-value {risk_class}">{risk_level}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    confidence = abs(churn_prob - threshold) * 100
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Model Confidence</div>
            <div class="metric-value">{confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ================= VISUAL FEEDBACK =================
st.subheader("📈 Prediction Insight")

st.progress(int(churn_prob * 100))

if churn_prob > threshold:
    st.error(
        "⚠️ **Customer is likely to churn.**  \n"
        "Recommended actions: retention offers, personalized communication, loyalty benefits."
    )
else:
    st.success(
        "✅ **Customer is unlikely to churn.**  \n"
        "Customer appears stable based on current behavioral and financial indicators."
    )

# ================= FOOTER =================
st.divider()
st.caption(
    "End-to-end ML project: Data preprocessing → ANN model → Real-time prediction → Business insight dashboard"
)
