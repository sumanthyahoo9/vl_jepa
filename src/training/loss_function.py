"""
InfoNCE Loss for VL-JEPA.

Implements contrastive learning loss that encourages predicted embeddings
to match their corresponding targets while pushing apart from other samples in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) Loss.
    
    For each sample i in a batch:
    - Positive pair: (predicted_i, target_i)
    - Negative pairs: (predicted_i, target_j) for all j != i
    
    The loss encourages high similarity for positive pairs and low similarity
    for negative pairs using a contrastive objective.
    
    Args:
        temperature: Temperature parameter for scaling similarities (default: 1.0)
        reduction: Reduction method - 'mean' or 'sum' (default: 'mean')
    """
    
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            predictions: Predicted embeddings [B, seq_len, embed_dim]
            targets: Target embeddings [B, seq_len, embed_dim]
        
        Returns:
            loss: Scalar loss value
        
        Shape:
            - predictions: [B, 512, 1536]
            - targets: [B, 512, 1536]
            - output: scalar
        """
        # Step 1: Aggregate sequence dimension via mean pooling
        # [B, 512, 1536] → [B, 1536]
        predictions = torch.mean(predictions, dim=1)
        targets = torch.mean(targets, dim=1)
        
        # Step 2: L2 normalize embeddings
        # This makes cosine similarity equivalent to dot product
        predictions_norm = F.normalize(predictions, p=2, dim=-1)
        targets_norm = F.normalize(targets, p=2, dim=-1)
        
        # Step 3: Compute similarity matrix [B, B]
        # similarity[i, j] = cosine_similarity(pred_i, target_j)
        # Diagonal contains positive pairs, off-diagonal contains negatives
        similarity_matrix = predictions_norm @ targets_norm.T
        
        # Step 4: Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # Step 5: Compute log probabilities using log-softmax
        # For each row i, this computes:
        # log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
        log_probs = F.log_softmax(similarity_matrix, dim=-1)
        
        # Step 6: Extract positive pairs (diagonal) and compute loss
        # We want to maximize log_prob of correct matches (minimize negative log_prob)
        positive_log_probs = torch.diag(log_probs)
        loss = -positive_log_probs
        
        # Step 7: Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


def info_nce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Functional interface for InfoNCE loss.
    
    Args:
        predictions: Predicted embeddings [B, seq_len, embed_dim]
        targets: Target embeddings [B, seq_len, embed_dim]
        temperature: Temperature parameter for scaling similarities
    
    Returns:
        loss: Scalar loss value
    
    Example:
        >>> predictions = torch.randn(8, 512, 1536)
        >>> targets = torch.randn(8, 512, 1536)
        >>> loss = info_nce_loss(predictions, targets)
    """
    loss_fn = InfoNCELoss(temperature=temperature)
    return loss_fn(predictions, targets)


if __name__ == "__main__":
    # Quick test
    print("Testing InfoNCE Loss...")
    
    # Create dummy data
    batch_size = 8
    seq_len = 512
    embed_dim = 1536
    
    predictions = torch.randn(batch_size, seq_len, embed_dim)
    targets = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test module interface
    loss_fn = InfoNCELoss(temperature=1.0)
    loss = loss_fn(predictions, targets)
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Loss shape: {loss.shape}")
    
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    
    # Test functional interface
    loss_functional = info_nce_loss(predictions, targets, temperature=0.5)
    print(f"Functional loss: {loss_functional.item():.4f}")
    
    # Test gradient flow
    predictions.requires_grad = True
    loss = loss_fn(predictions, targets)
    loss.backward()
    
    assert predictions.grad is not None, "Gradients should flow"
    print(f"Gradient shape: {predictions.grad.shape}")
    print(f"Gradient norm: {predictions.grad.norm().item():.4f}")
    
    print("\n✓ All tests passed!")