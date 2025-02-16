import numpy as np 

np.random.seed(0)
data = np.random.randn(1000, 7)
print(f"No. of samples : {data.shape[0]}")
print(f"No. of features : {data.shape[1]}")

data

def pca(X, n_components):
    # Center the data
    X_meaned = X - np.mean(X, axis=0)
   
    # Calculate the covariance matrix
    cov_matrix = np.cov(X_meaned, rowvar=False)
   
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
   
    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
   
    # Select the top n_components eigenvectors
    top_eigenvectors = sorted_eigenvectors[:, :n_components]
   
    # Project the data onto the new subspace
    reduced_data = np.dot(X_meaned, top_eigenvectors)
   
    return reduced_data, top_eigenvectors

reduced_data, principal_components = pca(data, n_components=3)

# Print the reduced dimensions and principal components
print("Original shape:", data.shape)
print("Reduced shape:", reduced_data.shape)
print("Reduced dimensions:")
print(reduced_data)
print("\nPrincipal components:")
print(principal_components)