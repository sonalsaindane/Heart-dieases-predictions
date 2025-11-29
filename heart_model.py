# heart_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\hdpr\Heart-Disease-Prediction\cleveland.csv", header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']

# ------------------------------
# 2. Map target and sex
# ------------------------------
df['target'] = df.target.map({0:0, 1:1, 2:1, 3:1, 4:1})
df['sex'] = df.sex.map({'female':0,'male':1})

# ------------------------------
# 3. Replace '?' with np.nan and convert to numeric
# ------------------------------
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ------------------------------
# 4. Impute missing values
# ------------------------------
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df.iloc[:, :-1].values)  # features
y = df.iloc[:, -1].values                           # target

# ------------------------------
# 5. Split & scale
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# 6. Train models
# ------------------------------
models = {
    "SVM": SVC(kernel='rbf', probability=True).fit(X_train, y_train),
    "Naive Bayes": GaussianNB().fit(X_train, y_train),
    "Logistic Regression": LogisticRegression(max_iter=1000).fit(X_train, y_train),
    "Decision Tree": DecisionTreeClassifier().fit(X_train, y_train),
    "Random Forest": RandomForestClassifier(n_estimators=100).fit(X_train, y_train),
    "LightGBM": lgb.train({}, lgb.Dataset(X_train, label=y_train), 100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
}

# ------------------------------
# 7. Prediction function
# ------------------------------
def predict(model_name, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    # Replace any non-numeric input with NaN
    input_array = np.where(input_array == '?', np.nan, input_array).astype(float)
    
    # Impute missing values
    input_array = imputer.transform(input_array)
    
    # Scale
    input_scaled = scaler.transform(input_array)
    
    # Predict
    if model_name == "LightGBM":
        pred = models[model_name].predict(input_scaled)
        return int(pred[0] >= 0.5)
    else:
        pred = models[model_name].predict(input_scaled)[0]
        return int(pred)
