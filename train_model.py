import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

print("=== ড্রাই আই ডিজিজ প্রেডিকশন মডেল ট্রেইনিং ===")
print("ডেটা লোড হচ্ছে...")


file_path = "Dry_Eye_Dataset.csv"
try:
    data = pd.read_csv(file_path)
    print(f"✅ ডেটা সফলভাবে লোড হয়েছে: {data.shape[0]} রেকর্ড, {data.shape[1]} কলাম")
except FileNotFoundError:
    print(f"❌ ত্রুটি: {file_path} ফাইল পাওয়া যাচ্ছে না")
    exit()


print("\nডেটা প্রিপ্রসেসিং...")
data[['Systolic_BP', 'Diastolic_BP']] = data['Blood pressure'].str.split('/', expand=True).astype(float)
data.drop('Blood pressure', axis=1, inplace=True)

column_order = [
    'Gender', 'Age', 'Sleep duration', 'Sleep quality', 'Stress level', 
    'Systolic_BP', 'Diastolic_BP', 'Heart rate', 'Daily steps', 'Physical activity', 
    'Height', 'Weight', 'Sleep disorder', 'Wake up during night', 'Feel sleepy during day', 
    'Caffeine consumption', 'Alcohol consumption', 'Smoking', 'Medical issue', 
    'Ongoing medication', 'Smart device before bed', 'Average screen time', 
    'Blue-light filter', 'Discomfort Eye-strain', 'Redness in eye', 
    'Itchiness/Irritation in eye', 'Dry Eye Disease'
]
data = data[column_order]

#Binary coloum encoding
binary_cols = [
    'Sleep disorder', 'Wake up during night', 'Feel sleepy during day', 
    'Caffeine consumption', 'Alcohol consumption', 'Smoking', 'Medical issue', 
    'Ongoing medication', 'Smart device before bed', 'Blue-light filter', 
    'Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye', 
    'Dry Eye Disease'
]
for col in binary_cols:
    data[col] = data[col].map({'Y': 1, 'N': 0})

# Gender encoding
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])


X = data.drop('Dry Eye Disease', axis=1)
y = data['Dry Eye Disease']

print("\n📊 ক্লাস ডিস্ট্রিবিউশন:")
print(f"   Dry Eye Disease না (0): {sum(y == 0)} রেকর্ড ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"   Dry Eye Disease হ্যাঁ (1): {sum(y == 1)} রেকর্ড ({sum(y == 1)/len(y)*100:.1f}%)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  
)

print(f"\n📁 ডেটা স্প্লিট:")
print(f"   ট্রেইনিং ডেটা: {X_train.shape[0]} রেকর্ড")
print(f"   টেস্ট ডেটা: {X_test.shape[0]} রেকর্ড")


print("\n🔄 ডেটা ইমব্যালেন্স সমাধান করা হচ্ছে (SMOTE)...")
print("   SMOTE প্রয়োগের আগে ট্রেইনিং ডেটা:")
print(f"   - না (0): {sum(y_train == 0)} রেকর্ড")
print(f"   - হ্যাঁ (1): {sum(y_train == 1)} রেকর্ড")

try:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("   ✅ SMOTE প্রয়োগ成功")
    print("   SMOTE প্রয়োগের পরে ট্রেইনিং ডেটা:")
    print(f"   - না (0): {sum(y_train == 0)} রেকর্ড")
    print(f"   - হ্যাঁ (1): {sum(y_train == 1)} রেকর্ড")
except Exception as e:
    print(f"   ❌ SMOTE失败: {e}")
    print("   ⚠️ SMOTE ছাড়া继续 করা হচ্ছে")


numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Gender' in numerical_cols:
    numerical_cols.remove('Gender')

print(f"\n⚖️ স্কেলিং করা হচ্ছে {len(numerical_cols)}টি সংখ্যাগত কলাম")
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


print("\n🤖 মডেল ট্রেইনিং...")
model = SVC(
    kernel='linear',
    C=1.0,
    class_weight='balanced',  
    random_state=42, 
    probability=True
)
model.fit(X_train, y_train)
print("   ✅ মডেল ট্রেইনিং সম্পন্ন")


print("\n📈 মডেল ইভ্যালুয়েশন:")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"   ট্রেইনিং একুরেসি: {accuracy_score(y_train, y_train_pred):.2%}")
print(f"   টেস্ট একুরেসি: {accuracy_score(y_test, y_test_pred):.2%}")

print("\n🔍 কনফিউশন ম্যাট্রিক্স (টেস্ট ডেটা):")
cm = confusion_matrix(y_test, y_test_pred)
print(f"   True Negative  (সঠিক 'না'): {cm[0][0]}")
print(f"   False Positive (ভুল 'হ্যাঁ'): {cm[0][1]}")
print(f"   False Negative (ভুল 'না'): {cm[1][0]}")
print(f"   True Positive  (সঠিক 'হ্যাঁ'): {cm[1][1]}")

print("\n📋 ক্লাসিফিকেশন রিপোর্ট:")
print(classification_report(y_test, y_test_pred))


print("\n🎯 বিভিন্ন থ্রেশহোল্ডে পারফরম্যান্স:")
y_test_proba = model.predict_proba(X_test)[:, 1]  

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("   থ্রেশহোল্ড | একুরেসি | TN/FP/FN/TP")
print("   " + "-" * 40)

for threshold in thresholds:
    y_pred_threshold = (y_test_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_threshold)
    cm = confusion_matrix(y_test, y_pred_threshold)
    print(f"   {threshold} | {accuracy:.2%} | {cm[0][0]}/{cm[0][1]}/{cm[1][0]}/{cm[1][1]}")


print(f"\n🧪 স্যাম্পল প্রেডিকশন টেস্ট (৫টি ডেটা):")
sample_data = X_test.iloc[:5]
sample_actual = y_test.iloc[:5]
sample_pred = model.predict(sample_data)
sample_proba = model.predict_proba(sample_data)

for i in range(len(sample_data)):
    actual = "হ্যাঁ" if sample_actual.iloc[i] == 1 else "না"
    predicted = "হ্যাঁ" if sample_pred[i] == 1 else "না"
    prob_no = sample_proba[i][0] * 100
    prob_yes = sample_proba[i][1] * 100
    print(f"   ডেটা {i+1}: আসল = {actual}, প্রেডিক্ট = {predicted}, সম্ভাবনা (না/হ্যাঁ) = {prob_no:.1f}%/{prob_yes:.1f}%")


print("\n💾 মডেল সেভ করা হচ্ছে...")
with open('svc_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('le_gender.pkl', 'wb') as f:
    pickle.dump(le_gender, f)

with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("✅ সব ফাইল সফলভাবে সেভ হয়েছে!")
print("\n🎉 মডেল ট্রেইনিং সম্পূর্ণ!")
print("💡 রিকমেন্ডেড থ্রেশহোল্ড: 0.4 বা 0.5")