import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Parameters
np.random.seed(42)
n_simulations = 1000
n_observations = 200
n_features = 10
k_folds = 5
n_components_range = range(1, n_features + 1)

# True coefficients
true_beta = np.random.uniform(-1, 1, size=n_features)

# Storage for OLS
ols_cv_mse_list = []
ols_y_actual_vs_predicted = []
estimated_betas_ols = []  # Store estimated coefficients for all folds

# Storage for PCR and PLS
pcr_coef_mse = {k: [] for k in n_components_range}
pcr_bias_squared = {k: [] for k in n_components_range}
pcr_variance = {k: [] for k in n_components_range}
pcr_estimated_betas = {k: [] for k in n_components_range}
pcr_cv_mse = {k: [] for k in n_components_range}

pls_coef_mse = {k: [] for k in n_components_range}
pls_bias_squared = {k: [] for k in n_components_range}
pls_variance = {k: [] for k in n_components_range}
pls_estimated_betas = {k: [] for k in n_components_range}
pls_cv_mse = {k: [] for k in n_components_range}

pcr_y_actual_vs_predicted = {k: [] for k in n_components_range}
pls_y_actual_vs_predicted = {k: [] for k in n_components_range}

# Monte Carlo Simulation
for sim in range(n_simulations):
    # Generate independent data (no multicollinearity)
    X = np.random.normal(0, 1, size=(n_observations, n_features))
    residuals = np.random.normal(0, 0.1, size=n_observations)
    y = X @ true_beta + residuals

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=sim)
    ols_fold_mse = []
    ols_fold_coefficients = []
    ols_fold_actual_vs_predicted = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # OLS
        ols_model = LinearRegression(fit_intercept=False)
        ols_model.fit(X_train, y_train)

        y_pred_ols = ols_model.predict(X_val)
        ols_fold_mse.append(mean_squared_error(y_val, y_pred_ols))
        ols_fold_coefficients.append(ols_model.coef_)
        ols_fold_actual_vs_predicted.append((y_val, y_pred_ols))

    ols_cv_mse_list.append(np.mean(ols_fold_mse))
    estimated_betas_ols.extend(ols_fold_coefficients)
    ols_y_actual_vs_predicted.extend(ols_fold_actual_vs_predicted)

    # PCR and PLS
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # PCR
        for n_components in n_components_range:
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_val_pca = pca.transform(X_val)

            pcr_model = LinearRegression()
            pcr_model.fit(X_train_pca, y_train)
            estimated_beta_pcr = pca.inverse_transform(pcr_model.coef_)
            pcr_estimated_betas[n_components].append(estimated_beta_pcr)

            y_pred_pcr = pcr_model.predict(X_val_pca)
            pcr_mse = mean_squared_error(y_val, y_pred_pcr)
            pcr_cv_mse[n_components].append(pcr_mse)

            # Store actual and predicted values for PCR
            pcr_y_actual_vs_predicted[n_components].append((y_val, y_pred_pcr))

        # PLS
        for n_components in n_components_range:
            pls_model = PLSRegression(n_components=n_components)
            pls_model.fit(X_train, y_train)

            estimated_beta_pls = pls_model.coef_.ravel()
            pls_estimated_betas[n_components].append(estimated_beta_pls)

            y_pred_pls = pls_model.predict(X_val)

            pls_mse = mean_squared_error(y_val, y_pred_pls)
            pls_cv_mse[n_components].append(pls_mse)

            # Store actual and predicted values for PLS
            pls_y_actual_vs_predicted[n_components].append((y_val, y_pred_pls))

# Convert estimated betas to arrays for easier processing
estimated_betas_ols = np.array(estimated_betas_ols)

for n_components in n_components_range:
    pcr_estimated_betas[n_components] = np.array(pcr_estimated_betas[n_components])
    pls_estimated_betas[n_components] = np.array(pls_estimated_betas[n_components])

# OLS CV MSE
average_ols_cv_mse = np.mean(ols_cv_mse_list)
print(f"Average OLS CV MSE: {average_ols_cv_mse}")

# OLS MSE decomposition for coefficients
mean_estimated_beta_ols = np.mean(estimated_betas_ols, axis=0)
ols_bias_squared = np.mean((mean_estimated_beta_ols - true_beta) ** 2)
ols_variance = np.mean(np.var(estimated_betas_ols, axis=0))
ols_mse = np.mean((estimated_betas_ols - true_beta) ** 2)
assert np.isclose(ols_mse, ols_bias_squared + ols_variance, atol=1e-6), "OLS MSE decomposition mismatch!"

# PCR and PLS MSE decomposition for coefficients
for n_components in n_components_range:
    # PCR
    mean_estimated_beta_pcr = np.mean(pcr_estimated_betas[n_components], axis=0)
    pcr_bias_squared[n_components] = np.mean((mean_estimated_beta_pcr - true_beta) ** 2)
    pcr_variance[n_components] = np.mean(np.var(pcr_estimated_betas[n_components], axis=0))
    pcr_coef_mse[n_components] = np.mean((pcr_estimated_betas[n_components] - true_beta) ** 2)
    assert np.isclose(
        pcr_coef_mse[n_components],
        pcr_bias_squared[n_components] + pcr_variance[n_components],
        atol=1e-6
    ), f"PCR MSE decomposition mismatch for {n_components} components!"

    # PLS
    mean_estimated_beta_pls = np.mean(pls_estimated_betas[n_components], axis=0)
    pls_bias_squared[n_components] = np.mean((mean_estimated_beta_pls - true_beta) ** 2)
    pls_variance[n_components] = np.mean(np.var(pls_estimated_betas[n_components], axis=0))
    pls_coef_mse[n_components] = np.mean((pls_estimated_betas[n_components] - true_beta) ** 2)
    assert np.isclose(
        pls_coef_mse[n_components],
        pls_bias_squared[n_components] + pls_variance[n_components],
        atol=1e-6
    ), f"PLS MSE decomposition mismatch for {n_components} components!"

# Average MSE of predicted Y across folds and simulations for PCR and PLS
avg_pcr_cv_mse = {k: np.mean(v) for k, v in pcr_cv_mse.items()}
avg_pls_cv_mse = {k: np.mean(v) for k, v in pls_cv_mse.items()}

# Unite folds for corresponding coefficients of PCR for each component
united_coefficients_pcr = {}

for num_components, fold_coeffs in pcr_estimated_betas.items():  # Iterate over each component
    # Stack all folds for the current number of components
    united_coefficients_pcr[num_components] = np.vstack(fold_coeffs)

# Unite folds for corresponding coefficients of PLS for each component
united_coefficients_pls = {}

for num_components, fold_coeffs in pls_estimated_betas.items():  # Iterate over each component
    # Stack all folds for the current number of components
    united_coefficients_pls[num_components] = np.vstack(fold_coeffs)

# Find the optimal number of components for PCR and PLS
optimal_pcr_components = min(avg_pcr_cv_mse, key=avg_pcr_cv_mse.get)
optimal_pls_components = min(avg_pls_cv_mse, key=avg_pls_cv_mse.get)

print(f"Optimal number of components for PCR: {optimal_pcr_components}")
print(f"Optimal number of components for PLS: {optimal_pls_components}")
print(f"Average PCR CV MSE: {avg_pcr_cv_mse[optimal_pcr_components]}")
print(f"Average PLS CV MSE: {avg_pls_cv_mse[optimal_pls_components]}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define components to plot for PCR and PLS
components_to_plot_pcr = [1, 5, n_features]
components_to_plot_pls = [1, 5, n_features]

# First plot: Coefficient Comparison for different components
for i, (n_components_pcr, n_components_pls) in enumerate(zip(components_to_plot_pcr, components_to_plot_pls)):
    ax = axes[i // 2, i % 2]
    ax.plot(true_beta, true_beta, 'r--', label='True Coefficients')
    if n_components_pcr in pcr_estimated_betas:
        avg_pcr_betas = np.mean(pcr_estimated_betas[n_components_pcr], axis=0)
        ax.plot(true_beta, avg_pcr_betas, 'o', label=f'PCR (Components={n_components_pcr})', color='blue')
    if n_components_pls in pls_estimated_betas:
        avg_pls_betas = np.mean(pls_estimated_betas[n_components_pls], axis=0)
        ax.plot(true_beta, avg_pls_betas, 's', label=f'PLS (Components={n_components_pls})', color='orange')
    ax.set_xlabel('True Coefficients')
    ax.set_ylabel('Estimated Coefficients')
    ax.set_title(f'Coefficient Comparison (PCR={n_components_pcr}, PLS={n_components_pls})')
    ax.legend()
    ax.grid(True)

# Second plot: Unbiasedness of OLS
ax = axes[1, 1]
ax.plot(true_beta, mean_estimated_beta_ols, 'o', label="Estimated vs True", color='purple')
ax.plot(true_beta, true_beta, 'r--', label="45-degree line (True)")
ax.set_xlabel("True Coefficients (Beta)")
ax.set_ylabel("Average Estimated Coefficients (Beta Hat)")
ax.set_title("Unbiasedness of OLS: Monte Carlo Simulation")
ax.legend()
ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
fig.savefig("First_plot.png")  # Save figure with only the two selected plots
plt.close(fig)

# Plotting Cross-Validation MSE of PCR and PLS vs. Number of Components
plt.figure(figsize=(12, 6))
plt.plot(list(avg_pcr_cv_mse.keys()), list(avg_pcr_cv_mse.values()), label='PCR CV MSE', marker='o')

plt.plot(list(avg_pls_cv_mse.keys()), list(avg_pls_cv_mse.values()), label='PLS CV MSE', marker='o', color='orange')

# OLS CV MSE (as a horizontal line)
plt.axhline(
    y=average_ols_cv_mse,
    color='red',
    linestyle='--',
    label='OLS CV MSE'
)

plt.title("Cross-Validated MSE vs Number of Components for PCR and PLS")
plt.xlabel("Number of Components")
plt.ylabel("Mean Squared Error ")
plt.legend()
plt.grid()

# Save the figure
plt.savefig("Second_plot.png", dpi=300, bbox_inches='tight')  # Adjust DPI and bbox_inches for quality and layout
plt.show()

# Define components to plot for PCR and PLS
components_to_plot_pcr = [1, 5, n_features]
components_to_plot_pls = [1, 5, n_features]

# Create a 1x4 grid for the plots
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

# Plot comparison for OLS
actual_ols = ols_y_actual_vs_predicted[0][0]  # Actual y from first simulation
predicted_ols = ols_y_actual_vs_predicted[0][1]  # Predicted y from first simulation
axes[0].scatter(actual_ols, predicted_ols, alpha=0.7, label="OLS Predictions")
axes[0].plot([min(actual_ols), max(actual_ols)], [min(actual_ols), max(actual_ols)], 'r--',
             label="Perfect Prediction Line")
axes[0].set_xlabel("Actual Y")
axes[0].set_ylabel("Fitted Y")
axes[0].set_title("Actual vs Fitted Y for OLS")
axes[0].legend()
axes[0].grid(True)

# Plot comparisons for PCR and PLS
for i, (n_components_pcr, n_components_pls) in enumerate(zip(components_to_plot_pcr, components_to_plot_pls)):
    actual_pcr, predicted_pcr = pcr_y_actual_vs_predicted[n_components_pcr][0]  # First fold of first simulation
    actual_pls, predicted_pls = pls_y_actual_vs_predicted[n_components_pls][0]  # First fold of first simulation
    axes[i + 1].scatter(actual_pcr, predicted_pcr, alpha=0.7, label=f"PCR (Components={n_components_pcr})",
                        color='blue')
    axes[i + 1].scatter(actual_pls, predicted_pls, alpha=0.7, label=f"PLS (Components={n_components_pls})",
                        color='orange')
    axes[i + 1].plot([min(actual_pcr), max(actual_pcr)], [min(actual_pcr), max(actual_pcr)], 'r--',
                     label="Perfect Prediction Line")
    axes[i + 1].set_xlabel("Actual Y")
    axes[i + 1].set_title(f"PCR & PLS (PCR={n_components_pcr}, PLS={n_components_pls})")
    axes[i + 1].legend()
    axes[i + 1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

fig.savefig("Third_plot.png")  # Save figure with only the two selected plots
plt.close(fig)

# Components to analyze of first coefficient for OLS, PLS and PCR
components_to_plot_pcr = [1, 5, n_features]
components_to_plot_pls = [1, 5, n_features]

# Create a figure for the grid of histograms
plt.figure(figsize=(20, 5))

# 1. Histogram for OLS
plt.subplot(1, 7, 1)
plt.hist(estimated_betas_ols[:, 0], bins=20, color='skyblue', alpha=0.8, edgecolor='black')
plt.axvline(true_beta[0], color='red', linestyle='--', label='True Coefficient')
plt.axvline(np.mean(estimated_betas_ols[:, 0]), color='blue', linestyle='-', label='Mean Coefficient')
plt.title("OLS")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# 2-4. Histograms for PCR (1, 5, 10 components)
for idx, num_components in enumerate(components_to_plot_pcr, start=2):  # Subplots 2, 3, 4
    coeffs = united_coefficients_pcr[num_components][:, 0]  # First coefficient for PCR
    plt.subplot(1, 7, idx)
    plt.hist(coeffs, bins=20, color='skyblue', alpha=0.8, edgecolor='black')
    plt.axvline(true_beta[0], color='red', linestyle='--', label='True Coefficient')
    plt.axvline(np.mean(coeffs), color='blue', linestyle='-', label='Mean Coefficient')
    plt.title(f"PCR ({num_components} Components)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

# 5-7. Histograms for PLS (1, 5, 10 components)
for idx, num_components in enumerate(components_to_plot_pls, start=5):  # Subplots 5, 6, 7
    coeffs = united_coefficients_pls[num_components][:, 0]  # First coefficient for PLS
    plt.subplot(1, 7, idx)
    plt.hist(coeffs, bins=20, color='skyblue', alpha=0.8, edgecolor='black')
    plt.axvline(true_beta[0], color='red', linestyle='--', label='True Coefficient')
    plt.axvline(np.mean(coeffs), color='blue', linestyle='-', label='Mean Coefficient')
    plt.title(f"PLS ({num_components} Components)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.suptitle("Histograms of First Coefficient (OLS, PCR, PLS)", fontsize=16, y=1.05)
# Save the figure
plt.savefig("Fourth_plot.png", dpi=300, bbox_inches='tight')  # Adjust DPI and bbox_inches for quality and layout

plt.show()


# Define components to plot for PCR and PLS
components_to_plot_pcr = [1, 5, n_features]
components_to_plot_pls = [1, 5, n_features]

# Update methods and values for the plot
methods = ['OLS'] + \
          [f'PCR ({n})' for n in components_to_plot_pcr] + \
          [f'PLS ({n})' for n in components_to_plot_pls]

mse_values = [
    ols_mse,
    *[pcr_coef_mse[n] for n in components_to_plot_pcr],
    *[pls_coef_mse[n] for n in components_to_plot_pls]
]

bias_squared_values = [
    ols_bias_squared,
    *[pcr_bias_squared[n] for n in components_to_plot_pcr],
    *[pls_bias_squared[n] for n in components_to_plot_pls]
]

variance_values = [
    ols_variance,
    *[pcr_variance[n] for n in components_to_plot_pcr],
    *[pls_variance[n] for n in components_to_plot_pls]
]

# Create the bar chart
x = np.arange(len(methods))  # Adjust x to be a numpy array for proper bar alignment
width = 0.2  # Narrower width for better separation

plt.bar(x - width, mse_values, width=width, label='MSE', align='center')
plt.bar(x, bias_squared_values, width=width, label='Bias²', align='center')
plt.bar(x + width, variance_values, width=width, label='Variance', align='center', alpha=0.75)

# Add labels and title
plt.xticks(x, methods, rotation=45)
plt.ylabel('Value')
plt.title('MSE Decomposition for Coefficients (OLS, PCR, PLS)')
plt.legend()
plt.tight_layout()
# Save the figure
plt.savefig("Fifth_plot.png", dpi=300, bbox_inches='tight')  # Adjust DPI and bbox_inches for quality and layout

plt.show()




actual_pcr, predicted_pcr = pcr_y_actual_vs_predicted[optimal_pcr_components][0]  # First fold of first simulation
actual_pls, predicted_pls = pls_y_actual_vs_predicted[optimal_pls_components][0]  # First fold of first simulation
# Plot actual vs. predicted Y for OLS, PCR, and PLS in a single plot

plt.figure(figsize=(12, 6))

# OLS
plt.scatter(range(len(actual_ols)), actual_ols, label='OLS: Real Y', alpha=0.7, marker='o')
plt.scatter(range(len(predicted_ols)), predicted_ols, label='OLS: Predicted Y', alpha=0.7, marker='x', color='red')

# PCR
plt.scatter(range(len(actual_pcr)), actual_pcr, label=f'PCR (Components={optimal_pcr_components}): Real Y', alpha=0.7, marker='o', color='green')
plt.scatter(range(len(predicted_pcr)), predicted_pcr, label=f'PCR (Components={optimal_pcr_components}): Predicted Y', alpha=0.7, marker='x', color='orange')

# PLS
plt.scatter(range(len(actual_pls)), actual_pls, label=f'PLS (Components={optimal_pls_components}): Real Y', alpha=0.7, marker='o', color='blue')
plt.scatter(range(len(predicted_pls)), predicted_pls, label=f'PLS (Components={optimal_pls_components}): Predicted Y', alpha=0.7, marker='x', color='purple')

# Title and labels
plt.title("Real vs Predicted Y for OLS, PCR, and PLS")
plt.xlabel("Observation Index")
plt.ylabel("Y")
plt.legend()
plt.grid()

# Save the figure
plt.savefig("Sixth_plot.png", dpi=300, bbox_inches='tight')  # Adjust DPI and bbox_inches for quality and layout

# Show the plot
plt.show()



# Plot actual vs. predicted Y for OLS, PCR, and PLS as lines in 3 subplots (one grid)

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# OLS
axes[0].plot(range(len(actual_ols)), actual_ols, label='OLS: Real Y', linestyle='-', linewidth=1.5)
axes[0].plot(range(len(predicted_ols)), predicted_ols, label='OLS: Predicted Y', linestyle='--', linewidth=1.5, color='red')
axes[0].set_title("OLS: Real vs Predicted Y")
axes[0].set_ylabel("Y")
axes[0].legend()
axes[0].grid()

# PCR
axes[1].plot(range(len(actual_pcr)), actual_pcr, label=f'PCR (Components={optimal_pcr_components}): Real Y', linestyle='-', linewidth=1.5, color='green')
axes[1].plot(range(len(predicted_pcr)), predicted_pcr, label=f'PCR (Components={optimal_pcr_components}): Predicted Y', linestyle='--', linewidth=1.5, color='orange')
axes[1].set_title(f"PCR (Components={optimal_pcr_components}): Real vs Predicted Y")
axes[1].set_ylabel("Y")
axes[1].legend()
axes[1].grid()

# PLS
axes[2].plot(range(len(actual_pls)), actual_pls, label=f'PLS (Components={optimal_pls_components}): Real Y', linestyle='-', linewidth=1.5, color='blue')
axes[2].plot(range(len(predicted_pls)), predicted_pls, label=f'PLS (Components={optimal_pls_components}): Predicted Y', linestyle='--', linewidth=1.5, color='purple')
axes[2].set_title(f"PLS (Components={optimal_pls_components}): Real vs Predicted Y")
axes[2].set_xlabel("Observation Index")
axes[2].set_ylabel("Y")
axes[2].legend()
axes[2].grid()

# Adjust layout and display
plt.tight_layout()

# Save the figure
plt.savefig("Seventh_plot.png", dpi=300, bbox_inches='tight')  # Adjust DPI and bbox_inches for quality and layout

plt.show()


print(f"MSE OLS: {ols_mse}")
print(f"Bias² OLS: {ols_bias_squared}")
print(f"Variance OLS: {ols_variance}")
print(f"MSE PCR at optimal component: {pcr_coef_mse[optimal_pcr_components]}")
print(f"Bias² PCR at optimal component: {pcr_bias_squared[optimal_pcr_components]}")
print(f"Variance PCR at optimal component: {pcr_variance[optimal_pcr_components]}")
print(f"MSE PLS at optimal component: {pls_coef_mse[optimal_pls_components]}")
print(f"Bias² PLS at optimal component: {pls_bias_squared[optimal_pls_components]}")
print(f"Variance PLS at optimal component: {pls_variance[optimal_pls_components]}")