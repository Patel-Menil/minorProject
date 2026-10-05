"""
graph_builder.py
----------------
Step 4: Intra-hospital Patient Graph Construction

Dynamically constructs a patient-to-patient graph (adjacency matrix or
edge_index) from a batch of unified patient embeddings. Edges are drawn
between patients with similar clinical profiles.
"""

import torch
import torch.nn.functional as F


def build_knn_graph(
    embeddings: torch.Tensor,
    k: int = 5,
    metric: str = "cosine"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs a k-Nearest Neighbors graph from a batch of patient embeddings.
    
    Because the batch only contains patients from ONE hospital (due to the FL
    partitioning), this inherently enforces the rule that no cross-hospital
    edges are created.

    Parameters
    ----------
    embeddings : FloatTensor of shape (num_patients, fused_dim)
        The unified patient embeddings.
    k : int
        Number of nearest neighbors to connect to each patient.
    metric : str
        Similarity metric. Currently supports "cosine" or "euclidean".

    Returns
    -------
    edge_index : LongTensor of shape (2, num_edges)
        Standard PyTorch Geometric format for graph edges.
    edge_weight : FloatTensor of shape (num_edges,)
        The similarity scores for each edge.
    """
    num_patients = embeddings.size(0)
    
    # If there are fewer patients in the batch than 'k', cap 'k'
    k_actual = min(k, num_patients - 1)
    
    if k_actual <= 0:
        # Not enough patients to form edges
        return torch.empty((2, 0), dtype=torch.long, device=embeddings.device), \
               torch.empty((0,), dtype=torch.float32, device=embeddings.device)

    if metric == "cosine":
        # Normalize embeddings to compute cosine similarity via dot product
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.mm(norm_emb, norm_emb.t())  # (N, N)
        
        # Mask out self-loops with a large negative number so they aren't picked as neighbors
        similarity.fill_diagonal_(-1e9)
        
        # Get top-k nearest neighbors
        scores, indices = torch.topk(similarity, k=k_actual, dim=1, largest=True)
        
    elif metric == "euclidean":
        # Compute pairwise euclidean distances
        dist = torch.cdist(embeddings, embeddings, p=2)  # (N, N)
        
        # Mask out self-loops with a large positive number
        dist.fill_diagonal_(1e9)
        
        # Get bottom-k distances (nearest)
        distances, indices = torch.topk(dist, k=k_actual, dim=1, largest=False)
        # Convert distance to a similarity score (e.g., Gaussian kernel)
        scores = torch.exp(-distances)
        
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Build the edge_index array
    source_nodes = torch.arange(num_patients, device=embeddings.device).view(-1, 1).expand(-1, k_actual).flatten()
    target_nodes = indices.flatten()
    
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    edge_weight = scores.flatten()

    return edge_index, edge_weight


def build_threshold_graph(
    embeddings: torch.Tensor,
    threshold: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs a graph by connecting any two patients whose cosine similarity
    is strictly greater than the threshold.

    Parameters
    ----------
    embeddings : FloatTensor of shape (num_patients, fused_dim)
    threshold : float
        Minimum cosine similarity required to create an edge.

    Returns
    -------
    edge_index : LongTensor of shape (2, num_edges)
    edge_weight : FloatTensor of shape (num_edges,)
    """
    num_patients = embeddings.size(0)
    if num_patients < 2:
        return torch.empty((2, 0), dtype=torch.long, device=embeddings.device), \
               torch.empty((0,), dtype=torch.float32, device=embeddings.device)

    norm_emb = F.normalize(embeddings, p=2, dim=1)
    similarity = torch.mm(norm_emb, norm_emb.t())
    
    # Remove self loops
    similarity.fill_diagonal_(-1e9)
    
    # Find all pairs above threshold
    edges = (similarity > threshold).nonzero(as_tuple=False).t()
    
    if edges.size(1) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=embeddings.device), \
               torch.empty((0,), dtype=torch.float32, device=embeddings.device)
               
    # Extract weights for the created edges
    weights = similarity[edges[0], edges[1]]
    
    return edges, weights
