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

# Combined Grid for PCR
fig_pcr, axs_pcr = plt.subplots(2, 2, figsize=(16, 12))

# PCR: Cumulative Explained Variance Plot
axs_pcr[0, 0].plot(range(1, 6), explained_variance_pcr, marker='o', linestyle='-', color='b', label="Cumulative Explained Variance")
axs_pcr[0, 0].bar(range(1, 6), variance_ratios, alpha=0.7, color='orange', label="Variance per Component")
axs_pcr[0, 0].set_xticks(range(1, 6))
axs_pcr[0, 0].set_xlabel("Number of Components")
axs_pcr[0, 0].set_ylabel("Explained Variance (Ratio)")
axs_pcr[0, 0].set_title("PCR: Cumulative Explained Variance by Components")
axs_pcr[0, 0].legend()
axs_pcr[0, 0].grid(True)

# PCR: Variance Heatmap
sns.heatmap(covariance_matrix_pcr, annot=True, fmt=".2f", cmap="Blues", xticklabels=[f"PC{i+1}" for i in range(5)],
            yticklabels=[f"PC{i+1}" for i in range(5)], cbar=True, ax=axs_pcr[0, 1])
axs_pcr[0, 1].set_title("PCR: Variance Heatmap for Principal Components")
axs_pcr[0, 1].set_xlabel("Principal Components")
axs_pcr[0, 1].set_ylabel("Principal Components")

# PCR: Orthonormal Vectors in 2D
X_pca_2d = X_pca_pcr[:, :2]  # First 2 PCs for 2D visualization
W_pca_2d = pca.components_[:2].T / np.linalg.norm(pca.components_[:2].T, axis=0)  # Normalize PC directions
axs_pcr[1, 0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.7, label="Data Cloud (PCR)", color="lightblue")
for i, vector in enumerate(W_pca_2d.T):
    axs_pcr[1, 0].quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=f'C{i}', label=f'PC {i+1}')
axs_pcr[1, 0].axis('equal')
axs_pcr[1, 0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axs_pcr[1, 0].axvline(0, color='gray', linestyle='--', linewidth=0.5)
axs_pcr[1, 0].set_xlabel("First Principal Component")
axs_pcr[1, 0].set_ylabel("Second Principal Component")
axs_pcr[1, 0].set_title("")
axs_pcr[1, 0].legend()
axs_pcr[1, 0].grid(True)

# PCR: Orthonormal Vectors in 3D
ax_3d_pcr = fig_pcr.add_subplot(2, 2, 4, projection='3d')
X_pca_3d = X_pca_pcr[:, :3]  # First 3 PCs for 3D visualization
W_pca_3d = pca.components_[:3].T / np.linalg.norm(pca.components_[:3].T, axis=0)
ax_3d_pcr.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], alpha=0.7, color="lightblue", label="Data Cloud (PCR)")
for i, vector in enumerate(W_pca_3d.T):
    ax_3d_pcr.quiver(0, 0, 0, vector[0], vector[1], vector[2], length=1, color=f'C{i}', label=f'PC {i+1}', arrow_length_ratio=0.1)
ax_3d_pcr.set_xlabel("First Principal Component")
ax_3d_pcr.set_ylabel("Second Principal Component")
ax_3d_pcr.set_zlabel("Third Principal Component")
ax_3d_pcr.set_title("")
ax_3d_pcr.legend()

plt.tight_layout()
plt.show()

fig_pcr.savefig("PCR_Analysis.png")  # Save figure
plt.close(fig_pcr)

# Combined Grid for PLS
fig_pls, axs_pls = plt.subplots(2, 2, figsize=(16, 12))

# PLS: Covariance Maximization Plot
axs_pls[0, 0].bar(range(1, 6), np.abs(covariances_pls), alpha=0.7, label="Covariance per Component")
axs_pls[0, 0].plot(range(1, 6), cumulative_covariances_pls, marker='o', color='r', label="Cumulative Covariance")
axs_pls[0, 0].set_xticks(range(1, 6))
axs_pls[0, 0].set_xlabel("Number of Components")
axs_pls[0, 0].set_ylabel("Covariance")
axs_pls[0, 0].set_title("PLS: Covariance Maximization by Components")
axs_pls[0, 0].legend()
axs_pls[0, 0].grid(True)

# PLS: Covariance Heatmap
cov_matrix_pls = np.cov(np.hstack([X_scores_pls, y]), rowvar=False)
labels_pls = [f"Latent Comp {i+1}" for i in range(5)] + ["y"]
sns.heatmap(cov_matrix_pls, annot=True, fmt=".2f", xticklabels=labels_pls, yticklabels=labels_pls, cmap="coolwarm", cbar=True, ax=axs_pls[0, 1])
axs_pls[0, 1].set_title("PLS: Covariance Heatmap")


# PLS: Orthonormal Vectors in 2D
X_scores_2d_pls = X_scores_pls[:, :2]  # First 2 latent components for PLS
W_2d_pls = X_weights_pls[:, :2] / np.linalg.norm(X_weights_pls[:, :2], axis=0)
axs_pls[1, 0].scatter(X_scores_2d_pls[:, 0], X_scores_2d_pls[:, 1], alpha=0.7, label="Data Cloud (PLS)", color="lightgreen")
for i, vector in enumerate(W_2d_pls.T):
    axs_pls[1, 0].quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=f'C{i}', label=f'Latent Comp {i+1}')
axs_pls[1, 0].axis('equal')
axs_pls[1, 0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axs_pls[1, 0].axvline(0, color='gray', linestyle='--', linewidth=0.5)
axs_pls[1, 0].set_xlabel("First Latent Component")
axs_pls[1, 0].set_ylabel("Second Latent Component")
axs_pls[1, 0].set_title("")
axs_pls[1, 0].legend()
axs_pls[1, 0].grid(True)

# PLS: Orthonormal Vectors in 3D
ax_3d_pls = fig_pls.add_subplot(2, 2, 4, projection='3d')
X_scores_3d_pls = X_scores_pls[:, :3]  # First 3 latent components for PLS
W_3d_pls = X_weights_pls[:, :3] / np.linalg.norm(X_weights_pls[:, :3], axis=0)
ax_3d_pls.scatter(X_scores_3d_pls[:, 0], X_scores_3d_pls[:, 1], X_scores_3d_pls[:, 2], alpha=0.7, color="lightgreen", label="Data Cloud (PLS)")
for i, vector in enumerate(W_3d_pls.T):
    ax_3d_pls.quiver(0, 0, 0, vector[0], vector[1], vector[2], length=1, color=f'C{i}', label=f'Latent Comp {i+1}', arrow_length_ratio=0.1)
ax_3d_pls.set_xlabel("First Latent Component")
ax_3d_pls.set_ylabel("Second Latent Component")
ax_3d_pls.set_zlabel("Third Latent Component")
ax_3d_pls.set_title("")
ax_3d_pls.legend()

plt.tight_layout()
plt.show()

fig_pls.savefig("PLS_Analysis.png")  # Save figure
plt.close(fig_pls)