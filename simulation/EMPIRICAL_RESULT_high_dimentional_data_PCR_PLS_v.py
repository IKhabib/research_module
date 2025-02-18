import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to compute RMSE
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# PCR implementation
def perform_pcr(X_train, X_test, y_train, y_test, n_components):
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Fit Linear Regression on PCA components
    lr = LinearRegression()
    lr.fit(X_train_pca, y_train)

    # Predict and evaluate
    y_pred_test = lr.predict(X_test_pca)
    y_pred_train = lr.predict(X_train_pca)
    rmse_test = compute_rmse(y_test, y_pred_test)
    rmse_train = compute_rmse(y_train, y_pred_train)
    return rmse_test, rmse_train, pca, lr, y_pred_test

# PLS implementation
def perform_pls(X_train, X_test, y_train, y_test, n_components):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_test = pls.predict(X_test).flatten()
    y_pred_train = pls.predict(X_train).flatten()
    rmse_test = compute_rmse(y_test, y_pred_test)
    rmse_train = compute_rmse(y_train, y_pred_train)
    return rmse_test, rmse_train, pls, y_pred_test

# Load data
data_mp5spec = pd.read_csv("data_mp5spec.csv")
y_data = pd.read_csv("y_data.csv")

# Extract moisture column from y_data
y_moisture = y_data.iloc[:, 0]

# Align data_mp5spec and y_moisture
assert data_mp5spec.shape[0] == y_moisture.shape[0], "Mismatch in sample count between spectra and target data."

# Convert data to NumPy arrays for modeling
X = data_mp5spec.values
y = y_moisture.values

# Plot the original spectra of the corn samples
plt.figure(figsize=(10, 6))
plt.plot(X.T, alpha=0.7)
plt.xlabel("Wavelength Index", fontsize=12)
plt.ylabel("Absorbance", fontsize=12)
plt.title("Original NIR Spectra of 80 Corn Samples", fontsize=14)
plt.grid(True)
plt.show()


# Standardize the spectral data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Test with a range of components
max_components = 700  # Testing up to 700 components
results = {"Components": [], "PCR_RMSE_Test": [], "PLS_RMSE_Test": [], "PCR_RMSE_Train": [], "PLS_RMSE_Train": []}

for n in range(1, min(max_components, 25) + 1):  # Display RMSE for up to 25 components
    # PCR
    pcr_rmse_test, pcr_rmse_train, _, _, _ = perform_pcr(X_train, X_test, y_train, y_test, n)

    # PLS
    pls_rmse_test, pls_rmse_train, _, _ = perform_pls(X_train, X_test, y_train, y_test, n)

    # Store results
    results["Components"].append(n)
    results["PCR_RMSE_Test"].append(pcr_rmse_test)
    results["PLS_RMSE_Test"].append(pls_rmse_test)
    results["PCR_RMSE_Train"].append(pcr_rmse_train)
    results["PLS_RMSE_Train"].append(pls_rmse_train)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Plot RMSE vs. Number of Components for Training Data
plt.figure(figsize=(10, 6))
plt.plot(results_df["Components"], results_df["PCR_RMSE_Train"], label="PCR RMSE (Train)", marker='o', linestyle='--')
plt.plot(results_df["Components"], results_df["PLS_RMSE_Train"], label="PLS RMSE (Train)", marker='s', linestyle='-')
plt.xlabel("Number of Components", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title("RMSE Comparison of PCR and PLS on Training Data (Up to 25 Components)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Plot RMSE vs. Number of Components for Test Data
plt.figure(figsize=(10, 6))
plt.plot(results_df["Components"], results_df["PCR_RMSE_Test"], label="PCR RMSE (Test)", marker='o', linestyle='--')
plt.plot(results_df["Components"], results_df["PLS_RMSE_Test"], label="PLS RMSE (Test)", marker='s', linestyle='-')
plt.xlabel("Number of Components", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title("RMSE Comparison of PCR and PLS on Test Data (Up to 25 Components)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Compute metrics for optimal components
optimal_components_pcr = results_df.loc[results_df["PCR_RMSE_Test"].idxmin(), "Components"]
optimal_components_pls = results_df.loc[results_df["PLS_RMSE_Test"].idxmin(), "Components"]

# Refit PCR and PLS with optimal components
pcr_rmse_test, _, pca_model, lr_model, y_pred_pcr = perform_pcr(X_train, X_test, y_train, y_test, optimal_components_pcr)
pls_rmse_test, _, pls_model, y_pred_pls = perform_pls(X_train, X_test, y_train, y_test, optimal_components_pls)

# PCR regression coefficients (projected back to original space)
pcr_coefficients = np.dot(pca_model.components_.T, lr_model.coef_)

# Compute |Beta|^2 for optimal components coefficients
beta_squared_pcr = np.sum(pcr_coefficients ** 2)
beta_squared_pls = np.sum(pls_model.coef_.flatten() ** 2)

# Compute RDR metric
rdr_metric = np.linalg.norm(pcr_coefficients - pls_model.coef_.flatten()) / np.linalg.norm(pls_model.coef_.flatten())

# Metrics table
metrics_table = pd.DataFrame({
    "Method": ["PCR", "PLS"],
    "Optimal Components": [optimal_components_pcr, optimal_components_pls],
    "RMSECV": [pcr_rmse_test, pls_rmse_test],
    "Correlation": [
        np.corrcoef(pcr_coefficients.flatten(), pls_model.coef_.flatten())[0, 1],
        np.corrcoef(pcr_coefficients.flatten(), pls_model.coef_.flatten())[0, 1]
    ]
})

# Display metrics table
print("Metrics Table")
print(metrics_table)

# Adjust wavelength range for plotting
wavelength_start = 1000
wavelength_end = 2400
wavelength_indices = np.linspace(wavelength_start, wavelength_end, len(pcr_coefficients), dtype=int)

# Plot the coefficients
plt.figure(figsize=(10, 6))
plt.plot(wavelength_indices, pcr_coefficients, label=f"PCR Coefficients (n={optimal_components_pcr})", linestyle='--')
plt.plot(wavelength_indices, pls_model.coef_.flatten(), label=f"PLS Coefficients (n={optimal_components_pls})", linestyle='-')
plt.xlabel("Wavelength (nm)", fontsize=12)
plt.ylabel("Coefficient Value", fontsize=12)
plt.title("Regression Coefficients: PCR vs. PLS (Wavelength 1000-2400 nm)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Plot combined fitted vs actual values for PCR and PLS
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_pcr, label="PCR", alpha=0.7, marker='o')
plt.scatter(y_test, y_pred_pls, label="PLS", alpha=0.7, marker='s')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Fitted Values", fontsize=12)
plt.title("Fitted vs Actual Values: PCR and PLS", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
