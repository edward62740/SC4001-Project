import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from torch import Tensor

def compute_triplet_loss(embeddings, labels, margin=1.0):
    """
    returns the batch-hard triplet loss.
    embeddings: [B, D], CLS token embeddings from L-1
    labels: [B], class labels
    """
    N = embeddings.size(0)
    
    distances = torch.cdist(embeddings, embeddings, p=2)
 
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t())
    negative_mask = (labels != labels.t())

    diag_ind = torch.eye(N, dtype=torch.bool, device=embeddings.device)
    positive_mask = positive_mask & ~diag_ind  # dont compare with itself

    hardest_positive, _ = (distances * positive_mask.float()).max(dim=1)
    
    large_val = distances.max().item() + 1.0
    masked_negatives = distances + large_val * (~negative_mask).float()
    hardest_negative, _ = masked_negatives.min(dim=1)
    
    triplet_losses = F.relu(hardest_positive - hardest_negative + margin)
    
    valid_anchor = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
    if valid_anchor.sum() > 0:
        loss = triplet_losses[valid_anchor].mean()
    else:
        loss = torch.tensor(0.0, device=embeddings.device)
    
    return loss