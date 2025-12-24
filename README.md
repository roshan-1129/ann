Project Overview

Customer churn is one of the most critical challenges faced by businesses today.
This project uses an Artificial Neural Network (ANN) to predict whether a customer is likely to leave a company based on historical customer data.

A Streamlit web application is built on top of the trained ANN model so users can interactively enter customer details and get real-time churn predictions.

🚀 Key Features

End-to-end Machine Learning pipeline

ANN model built using TensorFlow / Keras

Data preprocessing using StandardScaler, LabelEncoder, and OneHotEncoder

Interactive Streamlit UI for real-time predictions

Model persistence using Pickle

Clean and modular project structure

🧠 Model Architecture

Input Layer (Customer features)

Multiple Hidden Dense Layers with ReLU activation

Output Layer with Sigmoid activation

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

🗂️ Project Structure
📁 customer-churn-ann
│
├── app.py                     # Streamlit application
├── ann_churn_model.h5         # Trained ANN model
├── scaler.pkl                 # StandardScaler object
├── label_encoder.pkl          # LabelEncoder object
├── onehot_encoder.pkl         # OneHotEncoder object
├── churn_model.ipynb          # Model training notebook
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation

📊 Input Features

The model takes the following customer attributes as input:

Geography

Gender

Credit Score

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

🖥️ Tech Stack

Python

TensorFlow / Keras

Pandas & NumPy

Scikit-learn

Streamlit

Pickle

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/customer-churn-ann.git
cd customer-churn-ann

2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py

🎯 Output

Churn = Yes → Customer is likely to leave

Churn = No → Customer is likely to stay

The prediction is generated in real time based on user input.

📈 Future Enhancements

Add model explainability (SHAP / LIME)

Improve UI with advanced Streamlit components

Deploy on Streamlit Cloud / AWS / Render

Add confidence score for predictions

👨‍💻 Author

Roshan S
Aspiring Full-Stack & AI Developer
🔗 LinkedIn: www.linkedin.com/in/roshan1129
📂 GitHub: https://github.com/roshan-1129
