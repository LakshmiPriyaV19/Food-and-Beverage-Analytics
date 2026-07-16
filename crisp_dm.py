import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# CRISP-DM Phase 1: Business Understanding
# Objective: Predict Beverage Category
# -----------------------------

# -----------------------------
# CRISP-DM Phase 2: Data Understanding
# -----------------------------
df = pd.read_csv("starbucks.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

# -----------------------------
# CRISP-DM Phase 3: Data Preparation
# -----------------------------
X = df.drop("Beverage_category", axis=1)
y = df["Beverage_category"]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# CRISP-DM Phase 4: Modeling
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# CRISP-DM Phase 5: Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("CRISP-DM Accuracy:", accuracy)

# -----------------------------
# CRISP-DM Phase 6: Deployment
# -----------------------------
# import joblib
# joblib.dump(model,"starbucks_model.pkl")
