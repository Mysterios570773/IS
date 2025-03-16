import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GaussianNoise
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ตั้งค่า Seed เพื่อให้ผลลัพธ์สม่ำเสมอ
np.random.seed(42)
tf.random.set_seed(42)

# โหลดข้อมูล
df = pd.read_csv("datasets/diabetes.csv")

# แยก Features และ Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# เลือก 8 Features ที่ดีที่สุด
k_best = SelectKBest(score_func=f_classif, k=8)
X_selected = k_best.fit_transform(X, y)

# บันทึก SelectKBest
joblib.dump(k_best, "models/diabetes_kbest.pkl")

# ลดมิติข้อมูลเหลือ 6 องค์ประกอบหลักด้วย PCA
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_selected)

# บันทึก PCA
joblib.dump(pca, "models/diabetes_pca.pkl")

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Scaling ข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# บันทึก StandardScaler
joblib.dump(scaler, "models/diabetes_scaler.pkl")

# ใช้ SMOTE เพื่อแก้ปัญหา class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ตั้งค่า Optimizer เป็น AdamW
optimizer = AdamW(learning_rate=0.003, weight_decay=1e-4)

# สร้างโมเดล Neural Network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    GaussianNoise(0.1),  # ลด Overfitting
    
    Dense(128, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# คอมไพล์โมเดล
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ใช้ Early Stopping และ ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)

# ฝึกโมเดล
history = model.fit(X_train_res, y_train_res, epochs=150, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# ประเมินโมเดล
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# บันทึกโมเดล
model.save("models/diabetes_nn_model_v5.keras")
