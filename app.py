import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from pathlib import Path

# กำหนด base directory
base_dir = Path("/Users/naja/Documents/GitHub/IS")  # กำหนดให้ตรงกับที่ไฟล์โมเดลอยู่

# โหลดโมเดลด้วยพาธที่ถูกต้อง
ml_model_path = base_dir / "models" / "employee_attrition_model.pkl"
nn_model_path = base_dir / "models" / "diabetes_nn_model_v5.keras"

# ตรวจสอบพาธว่าไฟล์มีอยู่หรือไม่
print("Trying to load model from:", ml_model_path)

ml_model = None
nn_model = None

# โหลดโมเดล
try:
    ml_model = joblib.load(str(ml_model_path)) 
    nn_model = tf.keras.models.load_model(str(nn_model_path))
    print("Models loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    # แจ้งให้ผู้ใช้ทราบหากไม่สามารถโหลดโมเดลได้
    st.error(f"Error loading model: {e}")

# ถ้าโมเดลถูกโหลดแล้ว ใช้ในฟังก์ชันที่เกี่ยวข้อง
if ml_model:
    def show_ml_demo():
        # ตัวอย่างการใช้ ml_model
        input_data = np.array([1, 2, 3, 4, 5])  # ข้อมูลตัวอย่าง
        prediction = ml_model.predict(input_data.reshape(1, -1))  # ปรับขนาดข้อมูลให้เหมาะสมกับโมเดล
        st.write(f"Prediction: {prediction}")
else:
    st.warning("Machine learning model not loaded. Please check model files.")

# ฟังก์ชันสำหรับ Navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Data Preparation",
        "Model Development",
        "Machine Learning Demo",
        "Neural Network Demo"
    ])
    
    if page == "Data Preparation":
        show_data_preparation()
    elif page == "Model Development":
        show_model_development()
    elif page == "Machine Learning Demo":
        show_ml_demo()
    elif page == "Neural Network Demo":
        show_nn_demo()

# หน้า Data Preparation
def show_data_preparation():
    st.title("1. Data Preparation")
    st.write("### Dataset Sources")
    st.markdown("- **Pima Indians Diabetes Dataset** [(Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)")
    st.markdown("- **IBM HR Analytics Employee Attrition & Performance** [(Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)")

# หน้า Model Development 
def show_model_development(): 
    st.title("2. Model Development")
    
    st.write("### Machine Learning Model")
    st.markdown("""
    - **อัลกอริทึม**: Logistic Regression
    - **วัตถุประสงค์**: ใช้สำหรับการพยากรณ์ Employee Attrition (การลาออกของพนักงาน)
    - **ขั้นตอนการพัฒนา**:
        1. **การเตรียมข้อมูล**: จัดการ Missing Values, One-Hot Encoding, และการแปลงค่า Target เป็นตัวเลข
        2. **การแบ่งข้อมูล**: แบ่งข้อมูลเป็นชุดฝึก (Training Set) และชุดทดสอบ (Test Set)
        3. **การฝึกโมเดล**: ฝึกโมเดลด้วยชุดฝึกและประเมินประสิทธิภาพด้วยชุดทดสอบ
        4. **การปรับปรุงโมเดล**: ปรับ Hyperparameters เพื่อเพิ่มประสิทธิภาพของโมเดล
    - **ประสิทธิภาพของโมเดล**:
        - **Accuracy**: 88%
    """)
    
    st.write("### Neural Network Model")
    st.markdown("""
    - **อัลกอริทึม**: Deep Neural Network (DNN)
    - **วัตถุประสงค์**: ใช้สำหรับการพยากรณ์ Diabetes (โรคเบาหวาน)
    - **ขั้นตอนการพัฒนา**:
        1. **การเตรียมข้อมูล**: จัดการ Missing Values, ปรับสเกลข้อมูล, และการแบ่งข้อมูล
        2. **การออกแบบโครงสร้างเครือข่าย**: ใช้โครงสร้างหลายชั้น (Multiple Layers) เพื่อเรียนรู้ข้อมูล
        3. **การฝึกโมเดล**: ฝึกโมเดลด้วยชุดฝึกและประเมินประสิทธิภาพด้วยชุดทดสอบ
        4. **การปรับปรุงโมเดล**: ปรับ Hyperparameters และโครงสร้างเครือข่ายเพื่อเพิ่มประสิทธิภาพ
    - **ประสิทธิภาพของโมเดล**:
        - **Accuracy**: 75%
    """)

# หน้า Machine Learning Demo (Employee Attrition)
def show_ml_demo():
    st.title("3. Machine Learning Demo")
    st.write("### Employee Attrition Prediction")
    
    # ข้อมูลที่ผู้ใช้กรอก
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    overtime = st.selectbox("Over Time", ["No", "Yes"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    
    # สร้างอาร์เรย์ขนาด 1x47 และเติมค่าเริ่มต้นเป็น 0
    input_data = np.zeros((1, 47))
    
    # ใส่ค่าที่ผู้ใช้กรอก
    input_data[0][0] = age  # Age
    input_data[0][1] = 1 if overtime == "Yes" else 0  # OverTime (Yes=1, No=0)
    input_data[0][2] = job_level  # JobLevel
    input_data[0][3] = monthly_income  # MonthlyIncome
    
    # แปลงข้อมูล categorical เป็น numerical
    department_mapping = {"Sales": 0, "Research & Development": 1, "Human Resources": 2}
    education_mapping = {"Life Sciences": 0, "Medical": 1, "Marketing": 2, "Technical Degree": 3, "Other": 4}
    gender_mapping = {"Male": 0, "Female": 1}
    marital_mapping = {"Single": 0, "Married": 1, "Divorced": 2}
    business_travel_mapping = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    
    input_data[0][4] = department_mapping[department]  # Department
    input_data[0][5] = education_mapping[education_field]  # EducationField
    input_data[0][6] = gender_mapping[gender]  # Gender
    input_data[0][7] = marital_mapping[marital_status]  # MaritalStatus
    input_data[0][8] = business_travel_mapping[business_travel]  # BusinessTravel
    
    # ใช้ค่า default สำหรับ features ที่เหลือ (หากไม่ต้องการให้ผู้ใช้กรอก)
    input_data[0][9] = 500  # DailyRate (ค่า default)
    input_data[0][10] = 10  # DistanceFromHome (ค่า default)
    input_data[0][11] = 3  # Education (ค่า default)
    input_data[0][12] = 3  # EnvironmentSatisfaction (ค่า default)
    input_data[0][13] = 50  # HourlyRate (ค่า default)
    input_data[0][14] = 3  # JobInvolvement (ค่า default)
    input_data[0][15] = 3  # JobSatisfaction (ค่า default)
    input_data[0][16] = 5000  # MonthlyRate (ค่า default)
    input_data[0][17] = 2  # NumCompaniesWorked (ค่า default)
    input_data[0][18] = 15  # PercentSalaryHike (ค่า default)
    input_data[0][19] = 3  # PerformanceRating (ค่า default)
    input_data[0][20] = 3  # RelationshipSatisfaction (ค่า default)
    input_data[0][21] = 80  # StandardHours (ค่า default)
    input_data[0][22] = 1  # StockOptionLevel (ค่า default)
    input_data[0][23] = 10  # TotalWorkingYears (ค่า default)
    input_data[0][24] = 2  # TrainingTimesLastYear (ค่า default)
    input_data[0][25] = 3  # WorkLifeBalance (ค่า default)
    input_data[0][26] = 5  # YearsAtCompany (ค่า default)
    input_data[0][27] = 3  # YearsInCurrentRole (ค่า default)
    input_data[0][28] = 2  # YearsSinceLastPromotion (ค่า default)
    input_data[0][29] = 3  # YearsWithCurrManager (ค่า default)
    
    # ทำนายผล
    if st.button("Predict Attrition"):
        prediction = ml_model.predict(input_data)
        st.write(f"Result: {'Leave' if prediction[0] == 0 else 'Stay'}")

# หน้า Neural Network Demo (Diabetes Prediction)
def show_nn_demo():
    st.title("4. Neural Network Demo")
    st.write("### Diabetes Prediction")
    
    # ข้อมูลที่ผู้ใช้กรอก
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    
    # สร้างอาร์เรย์ขนาด 1x6 และใส่ค่าที่ผู้ใช้กรอก
    input_data_nn = np.array([[pregnancies, glucose, blood_pressure, bmi, diabetes_pedigree, age]])
    
    # ทำนายผล
    if st.button("Predict Diabetes"):
        prediction_nn = nn_model.predict(input_data_nn)
        st.write(f"Result: {'Diabetic' if prediction_nn[0][0] > 0.5 else 'Non-Diabetic'}")

if __name__ == "__main__":
    main()