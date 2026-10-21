import numpy as np

def incremental_pca(batches, n_components):
    # batches: list of batches of samples (list of list of list of floats)
    # n_components: int, number of principal components to return
    # Returns: list of lists, shape (n_components, n_features)
    mu0, N0, S0 = 0, 0, 0
    for batch in batches:
        batch = np.asarray(batch)
        n_sample, n_feature = batch.shape
        if N0 == 0:
            S0 = np.zeros((n_feature, n_feature))
        mu = batch.mean(axis=0, keepdims=True)
        batch_cen = batch - mu
        batch_cov = batch_cen.T @ batch_cen
        delta = mu - mu0
        N_new = N0 + n_sample
        S0 += batch_cov + (delta.T @ delta) * (N0 * n_sample / N_new)
        mu0 += delta * n_sample / N_new
        N0 = N_new
    sigma = S0/N0
    eigvals, eigvecs = np.linalg.eigh(sigma)
    idxs = np.argsort(eigvals)[::-1]
    eigvec_k = eigvecs[:,idxs[:n_components]].T
    for component in eigvec_k:
        idx = np.argmax(np.abs(component))
        if component[idx] < 0:
            component *= -1
    return eigvec_k.tolist()
        




