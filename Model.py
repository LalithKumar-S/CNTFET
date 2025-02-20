import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load Data
try:
    from google.colab import drive
    drive.mount('/content/drive')
    data_path = '/your datasheet path'
    df = pd.read_csv(data_path)
except ImportError:
    data_path = 'NCNFET.csv'
    df = pd.read_csv(data_path)
    print("Running locally, ensure 'NCNFET.csv' is in the same directory.")

# 2. Data Preprocessing
df.dropna(inplace=True)

numerical_features = ['N', 'M', 'VGS', 'VDD']
scaler = StandardScaler()

X = df[numerical_features]
y = df['ID'].values.ravel()  # Convert y to 1D array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)

X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)

# 3. Model Training with Hyperparameter Tuning (Using RandomizedSearchCV)
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

model = RandomForestRegressor(random_state=42)
n_iter = 20
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                    n_iter=n_iter, cv=5, scoring='neg_mean_squared_error',
                                    n_jobs=-1, random_state=42)

try:
    random_search.fit(X_train_scaled_df, y_train)
except KeyboardInterrupt:
    print("Random search interrupted. Saving intermediate results...")
    joblib.dump(random_search, 'random_search_checkpoint.pkl')
    print("Checkpoint saved.")
else:
    best_model = random_search.best_estimator_
    joblib.dump(best_model, 'ncnfet_model.pkl')
    print("Random search completed successfully. Best model saved.")

# 4. Model Evaluation
y_pred = best_model.predict(X_test_scaled_df)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2e}")
print(f"R-squared: {r2:.4f}")

# 5. User Input, Calculations, and Comparison
def calculate_from_formulas(N, M, VGS, VDD, VDS):
    DCNT = 78.3 * np.sqrt(N**2 + M**2 + N * M)

    if DCNT < 1e-6:
        print("Warning: DCNT is very small. Check input N.")
        VTH = 0.0
    else:
        VTH = 0.43 / DCNT

    mu_Cox_W_L = 2.43e-3

    if VDS < (VGS - VTH):
        ID = mu_Cox_W_L * [(VGS - VTH) * VDS - (VDS**2 / 2)]
    else:
        ID = (mu_Cox_W_L / 2) * (VGS - VTH)**2

    Power = VDD * ID
    return DCNT, VTH, ID, Power

def predict_from_model(input_data):
    scaled_input = scaler.transform(input_data[numerical_features])
    scaled_input_df = pd.DataFrame(scaled_input, columns=numerical_features, index=input_data.index)
    predicted_id = best_model.predict(scaled_input_df)
    return predicted_id

while True:
    try:
        N = float(input("Enter the value for N: "))
        VGS = float(input("Enter the value for VGS: "))
        VDD = float(input("Enter the value for VDD: "))
        VDS = float(input("Enter the value for VDS: "))  # VDS must be equal to VDD

        user_input = pd.DataFrame({'N': [N], 'M': [0], 'VGS': [VGS], 'VDD': [VDD]})

        DCNT_formula, VTH_formula, ID_formula, Power_formula = calculate_from_formulas(N, 0, VGS, VDD, VDS)
        predicted_id = predict_from_model(user_input)

        Power_ml = VDD * predicted_id[0]

        print("\n--- Results ---")
        print("Formula Calculations:")

        DCNT_formatted = f"{DCNT_formula * 1e-3:.6f}"  # Multiply by 10^3 for 10^-3 representation
        VTH_formatted = f"{VTH_formula * 1e3:.6f}"  # Multiply by 10^-3 for 10^3 representation
        DCNT_formatted = DCNT_formatted.rstrip('0').rstrip('.') if '.' in DCNT_formatted else DCNT_formatted
        VTH_formatted = VTH_formatted.rstrip('0').rstrip('.') if '.' in VTH_formatted else VTH_formatted

        print(f"• DCNT (x10^-3): {DCNT_formatted}") #Added units
        print(f"• VTH (x10^3): {VTH_formatted}") #Added units
        print(f"• ID: {ID_formula:.2e}")
        print(f"• Power: {Power_formula:.2e}")

        print("\nML Predictions:")
        print(f"• ID: {predicted_id[0]:.2e}")
        print(f"• Power: {Power_ml:.2e}")

        # Comparison
        print("\n--- Comparison ---")
        print(f"ID Difference: {abs(ID_formula - predicted_id[0]):.2e}")
        print(f"Power Difference: {abs(Power_formula - Power_ml):.2e}")

    except ValueError:
        print("Invalid input. Please enter numbers only.")

    another_prediction = input("Do you want to make another prediction? (yes/no): ").lower()
    if another_prediction != 'yes':
        break
