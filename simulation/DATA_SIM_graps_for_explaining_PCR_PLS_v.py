import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for X and y
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
coefficients = np.random.randn(n_features, 1)
y = X @ coefficients + 0.1 * np.random.randn(n_samples, 1)

# PART 1: PLS Analysis
# Perform PLS using NIPALS
pls = PLSRegression(n_components=5)
pls.fit(X, y)

# Covariance maximization in PLS
X_scores_pls = pls.x_scores_  # Latent components of X
X_weights_pls = pls.x_weights_[:, :5]  # Weight vectors for PLS
covariances_pls = [np.cov(X_scores_pls[:, i], y[:, 0])[0, 1] for i in range(5)]
cumulative_covariances_pls = np.cumsum(np.abs(covariances_pls))

# PART 2: PCR Analysis
# Perform PCA for PCR
pca = PCA(n_components=5)
X_pca_pcr = pca.fit_transform(X)  # PCA-transformed X

# Variance and covariance in PCR
explained_variance_pcr = np.cumsum(pca.explained_variance_ratio_)
variance_ratios = pca.explained_variance_ratio_  # Variance ratios for each PC
covariance_matrix_pcr = np.diag(variance_ratios)

# Function to check orthogonality
def check_orthogonality(vectors, labels):
    for i in range(vectors.shape[1]):
        for j in range(i + 1, vectors.shape[1]):
            dot_product = np.dot(vectors[:, i], vectors[:, j])
            print(f"Dot product between {labels[i]} and {labels[j]}: {dot_product:.2e}")

# Check orthogonality for PCR and PLS
print("\nChecking orthogonality for PCR Principal Components:")
check_orthogonality(pca.components_.T, [f"PC{i+1}" for i in range(5)])

print("\nChecking orthogonality for PLS Latent Components:")
check_orthogonality(X_weights_pls, [f"Latent Comp {i+1}" for i in range(5)])

# Create a new figure with only the required two graphs for PCR Analysis

fig_pcr_selected, axs_pcr_selected = plt.subplots(1, 2, figsize=(16, 6))

# PCR: Cumulative Explained Variance Plot
axs_pcr_selected[0].plot(range(1, 6), explained_variance_pcr, marker='o', linestyle='-', color='b', label="Cumulative Explained Variance")
axs_pcr_selected[0].bar(range(1, 6), variance_ratios, alpha=0.7, color='orange', label="Variance per Component")
axs_pcr_selected[0].set_xticks(range(1, 6))
axs_pcr_selected[0].set_xlabel("Number of Components")
axs_pcr_selected[0].set_ylabel("Explained Variance (Ratio)")
axs_pcr_selected[0].set_title("PCR: Cumulative Explained Variance by Components")
axs_pcr_selected[0].legend()
axs_pcr_selected[0].grid(True)

# PCR: Orthonormal Vectors in 3D
ax_3d_pcr_selected = fig_pcr_selected.add_subplot(1, 2, 2, projection='3d')
X_pca_3d = X_pca_pcr[:, :3]  # First 3 PCs for 3D visualization
W_pca_3d = pca.components_[:3].T / np.linalg.norm(pca.components_[:3].T, axis=0)
ax_3d_pcr_selected.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], alpha=0.7, color="lightblue", label="Data Cloud (PCR)")
for i, vector in enumerate(W_pca_3d.T):
    ax_3d_pcr_selected.quiver(0, 0, 0, vector[0], vector[1], vector[2], length=1, color=f'C{i}', label=f'PC {i+1}', arrow_length_ratio=0.1)
ax_3d_pcr_selected.set_xlabel("First Principal Component")
ax_3d_pcr_selected.set_ylabel("Second Principal Component")
ax_3d_pcr_selected.set_zlabel("Third Principal Component")
ax_3d_pcr_selected.set_title("PCR: Orthonormal Vectors in 3D")
ax_3d_pcr_selected.legend()

plt.tight_layout()
fig_pcr_selected.savefig("PCR_Selected_Analysis.png")  # Save figure with only the two selected plots
plt.close(fig_pcr_selected)

# Return the updated saved file path
["PCR_Selected_Analysis.png"]


# Create a new figure with only the required two graphs for PLS Analysis

fig_pls_selected, axs_pls_selected = plt.subplots(1, 2, figsize=(16, 6))

# PLS: Covariance Maximization Plot
axs_pls_selected[0].bar(range(1, 6), np.abs(covariances_pls), alpha=0.7, label="Covariance per Component")
axs_pls_selected[0].plot(range(1, 6), cumulative_covariances_pls, marker='o', color='r', label="Cumulative Covariance")
axs_pls_selected[0].set_xticks(range(1, 6))
axs_pls_selected[0].set_xlabel("Number of Components")
axs_pls_selected[0].set_ylabel("Covariance")
axs_pls_selected[0].set_title("PLS: Covariance Maximization by Components")
axs_pls_selected[0].legend()
axs_pls_selected[0].grid(True)

# PLS: Orthonormal Vectors in 3D
ax_3d_pls_selected = fig_pls_selected.add_subplot(1, 2, 2, projection='3d')
X_scores_3d_pls = X_scores_pls[:, :3]  # First 3 latent components for PLS
W_3d_pls = X_weights_pls[:, :3] / np.linalg.norm(X_weights_pls[:, :3], axis=0)
ax_3d_pls_selected.scatter(X_scores_3d_pls[:, 0], X_scores_3d_pls[:, 1], X_scores_3d_pls[:, 2], alpha=0.7, color="lightgreen", label="Data Cloud (PLS)")
for i, vector in enumerate(W_3d_pls.T):
    ax_3d_pls_selected.quiver(0, 0, 0, vector[0], vector[1], vector[2], length=1, color=f'C{i}', label=f'Latent Comp {i+1}', arrow_length_ratio=0.1)
ax_3d_pls_selected.set_xlabel("First Latent Component")
ax_3d_pls_selected.set_ylabel("Second Latent Component")
ax_3d_pls_selected.set_zlabel("Third Latent Component")
ax_3d_pls_selected.set_title("PLS: Orthonormal Vectors in 3D")
ax_3d_pls_selected.legend()

plt.tight_layout()
fig_pls_selected.savefig("PLS_Selected_Analysis.png")  # Save figure with only the two selected plots
plt.close(fig_pls_selected)

# Return the updated saved file path
["PLS_Selected_Analysis.png"]
