import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np

# โหลดข้อมูล
data = pd.read_csv("datasets/employee_attrition.csv")

# แยกคอลัมน์ Target (Attrition) ออกก่อนทำ One-Hot Encoding
y = data["Attrition"]  # Target column
X = data.drop("Attrition", axis=1)  # Features

# จัดการ Missing Values โดยเติมค่ากลางแทน
X.fillna(X.median(numeric_only=True), inplace=True)

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# แปลงค่า Target จาก String เป็นตัวเลข (No -> 0, Yes -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# บันทึก Label Encoder
joblib.dump(label_encoder, "models/label_encoder.pkl")

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับสเกลข้อมูลสำหรับ Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# บันทึก StandardScaler
joblib.dump(scaler, "models/scaler.pkl")

# พัฒนาโมเดล Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# พัฒนาโมเดล Logistic Regression
lr_model = LogisticRegression(max_iter=10000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# พัฒนาโมเดล XGBoost
xgb_model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

# เปรียบเทียบประสิทธิภาพของโมเดล
models = {
    "Random Forest": rf_accuracy,
    "Logistic Regression": lr_accuracy,
    "XGBoost": xgb_accuracy
}

# หาโมเดลที่ดีที่สุด
best_model_name = sorted(models.items(), key=lambda x: x[1], reverse=True)[0][0]
best_model_accuracy = models[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy: {best_model_accuracy * 100:.2f}%")

# บันทึกโมเดลที่ดีที่สุด
model_path = "models/employee_attrition_model.pkl"
if best_model_name == "Random Forest":
    joblib.dump(rf_model, model_path)
elif best_model_name == "Logistic Regression":
    joblib.dump(lr_model, model_path)
else:
    joblib.dump(xgb_model, model_path)
