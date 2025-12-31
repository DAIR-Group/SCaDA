import numpy as np 
from scipy.linalg import block_diag


import matplotlib.pyplot as plt
def generate_data_matrix_normal(n, q, delta, U, V, seed, shift = None, no_cluster=False):
    """
    X ~ MN_{n x q}(MU, I_n, sigma^2 I_q)
    MU has first n/3 rows = mu1, next n/3 = mu2, last n/3 = mu3.
    """
    np.random.seed(seed)
    m = n 

    mu1 = np.zeros(q) 
    mu2 = np.zeros(q)
    mu3 = np.zeros(q)
    if not no_cluster:
        mu2[0] = delta
        mu3[0] = delta/2
        mu3[1] = np.sqrt(3)/2 *delta 
    MU = np.vstack([
        np.tile(mu1, (m,1)),
        np.tile(mu2, (m,1)),
        np.tile(mu3, (m,1))
    ])
    if shift:
        MU = MU + shift

    noise = generate_matrix_normal(M=np.zeros((3*n,q)), U=U, V=V)
    X = MU + noise
    labels = np.array([1]*m + [2]*m + [3]*m)

    return X, labels, MU
def random_3_clusters(ns=100,nt=50, dim=2, delta: float = 10, cluster_std = np.array([0.25, 0.5, 1]),  correlated = False, rho = 0.2, seed=None, return_Sigma=False, no_cluster=False):
    rng = np.random.default_rng(seed)
    shift = 2

    Sigma_ns = np.diag(np.repeat(cluster_std**2, ns))
    Sigma_nt = np.diag(np.repeat(cluster_std**2, nt))
    if correlated:
        Sigma_pp = np.array([[rho**abs(i-j) for j in range(dim)] for i in range(dim)])
    else:
        Sigma_pp = np.eye(dim)
    Xs, ys, mus = generate_data_matrix_normal(n=ns, q=dim, delta=8, U=Sigma_ns, V=Sigma_pp, seed=seed, no_cluster=no_cluster,)
    Xt, yt, mut = generate_data_matrix_normal(n=nt, q=dim, delta=delta, U=Sigma_nt, V=Sigma_pp, seed=seed, shift = shift, no_cluster=no_cluster)
    if return_Sigma:
        Sigma_nn = block_diag(Sigma_ns, Sigma_nt)
        vec_Sigma = np.kron(Sigma_nn, Sigma_pp)

        return Xs, Xt, ys, yt, mus, mut, vec_Sigma
    return Xs, Xt, ys, yt, mus, mut, None


def generate_matrix_normal(M, U, V):
    """
    Generates a sample from MN(M, U, V)
    M: Mean matrix (n x q)
    U: Between-row covariance (n x n)
    V: Between-column covariance (q x q)
    """
    n, q = M.shape
    
    # 1. Generate standard normal noise Z (n x q)
    Z = np.random.standard_normal((n, q))
    
    # 2. Compute Cholesky decomposition of U and V
    # We use lower triangular matrices
    L_u = np.linalg.cholesky(U)
    L_v = np.linalg.cholesky(V)
    
    # 3. Transform the noise: X = M + L_u @ Z @ L_v.T
    X = M + L_u @ Z @ L_v.T
    
    return X
if __name__ == "__main__":

    
    # Example usage
    Xs, Xt, ys, yt, mus, mut, Sigma = random_3_clusters(dim=10, delta=5, ns=100, nt=50, cluster_std=np.array([1, 1, 1]), correlated=False, rho=0.8, seed=40, return_Sigma=True, no_cluster=True)
    # random_3_clusters(dim=10, delta=5, ns=100, nt=50, cluster_std=[0.5,1,2],seed=40)
    # print("X shape:", X.shape)
    # print("y shape:", y.shape)
    plt.scatter(Xs[:,0], Xs[:,1], c=ys, cmap="viridis", s=30, alpha=0.7)
    plt.axis("equal")
    plt.show()
