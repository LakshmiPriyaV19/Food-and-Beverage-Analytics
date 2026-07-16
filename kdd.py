import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# KDD Step 1: Selection
# -----------------------------
df = pd.read_csv("starbucks.csv")
df.columns = df.columns.str.strip()

# -----------------------------
# KDD Step 2: Preprocessing
# -----------------------------
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop("Beverage_category", axis=1)
y = df["Beverage_category"]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# -----------------------------
# KDD Step 3: Transformation
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# KDD Step 4: Data Mining
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# KDD Step 5: Interpretation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("KDD Accuracy:", accuracy * 100)
