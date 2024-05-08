from __future__ import annotations
import torch



def KMeans(x: torch.Tensor, K: int = 10, Niter: int = 10, c: torch.Tensor | None = None):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    if c is None:
        c = x[:K, :].clone()  # Simplistic initialization for the centroids

    # x_i = x.view(N, 1, D)  # (N, 1, D) samples
    # c_j = c.view(1, K, D)  # (1, K, D) centroids
    for _ in range(Niter):
        # Calculate distances between points and centroids
        distances = torch.cdist(x, c)

        # Find the closest cluster for each point
        cluster_assignments = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros_like(c).scatter_add_(0, cluster_assignments.unsqueeze(1).repeat(1, D), x)

        cluster_counts = torch.zeros(K, device=x.device).scatter_add_(
            0, cluster_assignments, torch.ones(N, device=x.device)
        )

        # Avoid division by zero
        non_empty_clusters = cluster_counts > 0
        c[non_empty_clusters] = new_centroids[non_empty_clusters] / cluster_counts[non_empty_clusters].unsqueeze(1)

    return cluster_assignments, c
