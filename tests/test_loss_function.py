"""
Unit tests for InfoNCE Loss.
"""

import pytest
import torch
import torch.nn.functional as F
from src.training.loss_function import InfoNCELoss, info_nce_loss


class TestInfoNCELossShapes:
    """Test InfoNCE loss shape handling."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_output_is_scalar(self, batch_size):
        """Test that loss output is a scalar for various batch sizes."""
        predictions = torch.randn(batch_size, 512, 1536)
        targets = torch.randn(batch_size, 512, 1536)
        
        loss_fn = InfoNCELoss()
        loss = loss_fn(predictions, targets)
        
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.shape == torch.Size([]), "Loss should have empty shape"
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        batch_size = 4
        seq_lengths = [128, 256, 512, 1024]
        
        for seq_len in seq_lengths:
            predictions = torch.randn(batch_size, seq_len, 1536)
            targets = torch.randn(batch_size, seq_len, 1536)
            
            loss_fn = InfoNCELoss()
            loss = loss_fn(predictions, targets)
            
            assert loss.ndim == 0


class TestInfoNCELossProperties:
    """Test mathematical properties of InfoNCE loss."""
    
    def test_loss_is_positive(self):
        """Test that loss is always positive."""
        predictions = torch.randn(8, 512, 1536)
        targets = torch.randn(8, 512, 1536)
        
        loss_fn = InfoNCELoss()
        loss = loss_fn(predictions, targets)
        
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    
    def test_perfect_match_low_loss(self):
        """Test that perfect matches result in lower loss."""
        batch_size = 8
        
        # Case 1: Random predictions and targets
        predictions_random = torch.randn(batch_size, 512, 1536)
        targets_random = torch.randn(batch_size, 512, 1536)
        
        # Case 2: Predictions = Targets (perfect match)
        predictions_perfect = torch.randn(batch_size, 512, 1536)
        targets_perfect = predictions_perfect.clone()
        
        loss_fn = InfoNCELoss()
        loss_random = loss_fn(predictions_random, targets_random)
        loss_perfect = loss_fn(predictions_perfect, targets_perfect)
        
        # Perfect match should have much lower loss
        assert loss_perfect < loss_random, \
            f"Perfect match loss ({loss_perfect:.4f}) should be < random loss ({loss_random:.4f})"
    
    def test_temperature_effect(self):
        """Test that temperature parameter affects loss value."""
        predictions = torch.randn(8, 512, 1536)
        targets = torch.randn(8, 512, 1536)
        
        loss_temp_1 = InfoNCELoss(temperature=1.0)(predictions, targets)
        loss_temp_0_5 = InfoNCELoss(temperature=0.5)(predictions, targets)
        loss_temp_2 = InfoNCELoss(temperature=2.0)(predictions, targets)
        
        # Different temperatures should produce different losses
        assert loss_temp_0_5 != loss_temp_1
        assert loss_temp_2 != loss_temp_1


class TestInfoNCELossGradients:
    """Test gradient flow through InfoNCE loss."""
    
    def test_gradients_flow(self):
        """Test that gradients flow through the loss."""
        predictions = torch.randn(8, 512, 1536, requires_grad=True)
        targets = torch.randn(8, 512, 1536)
        
        loss_fn = InfoNCELoss()
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None, "Gradients should flow to predictions"
        assert not torch.isnan(predictions.grad).any(), "Gradients should not be NaN"
        assert not torch.isinf(predictions.grad).any(), "Gradients should not be Inf"
    
    def test_gradient_magnitude(self):
        """Test that gradient magnitudes are reasonable."""
        predictions = torch.randn(8, 512, 1536, requires_grad=True)
        targets = torch.randn(8, 512, 1536)
        
        loss_fn = InfoNCELoss()
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        grad_norm = predictions.grad.norm().item()
        
        # Gradients should be finite and not too large
        assert grad_norm > 0, "Gradient norm should be positive"
        assert grad_norm < 1000, f"Gradient norm too large: {grad_norm}"


class TestInfoNCELossContrastive:
    """Test contrastive properties of InfoNCE."""
    
    def test_diagonal_elements_are_positive_pairs(self):
        """Test that similarity matrix diagonal contains positive pairs."""
        batch_size = 4
        predictions = torch.randn(batch_size, 512, 1536)
        targets = torch.randn(batch_size, 512, 1536)
        
        # Manually compute what the loss does
        pred_pooled = torch.mean(predictions, dim=1)
        target_pooled = torch.mean(targets, dim=1)
        
        pred_norm = F.normalize(pred_pooled, dim=-1)
        target_norm = F.normalize(target_pooled, dim=-1)
        
        sim_matrix = pred_norm @ target_norm.T
        
        # Diagonal should have similarities for positive pairs
        assert sim_matrix.shape == (batch_size, batch_size)
        
        # Check that we can extract diagonal
        positive_sims = torch.diag(sim_matrix)
        assert positive_sims.shape == (batch_size,)
    
    def test_batch_size_one_behavior(self):
        """Test behavior with batch size 1 (no negatives)."""
        predictions = torch.randn(1, 512, 1536)
        targets = torch.randn(1, 512, 1536)
        
        loss_fn = InfoNCELoss()
        loss = loss_fn(predictions, targets)
        
        # With batch size 1, only positive pair exists
        # Loss should still be computable
        assert loss.ndim == 0
        assert loss.item() >= 0


class TestInfoNCELossFunctional:
    """Test functional interface."""
    
    def test_functional_interface(self):
        """Test that functional interface works."""
        predictions = torch.randn(8, 512, 1536)
        targets = torch.randn(8, 512, 1536)
        
        loss = info_nce_loss(predictions, targets, temperature=0.7)
        
        assert loss.ndim == 0
        assert loss.item() > 0
    
    def test_functional_matches_module(self):
        """Test that functional interface matches module interface."""
        predictions = torch.randn(8, 512, 1536)
        targets = torch.randn(8, 512, 1536)
        
        loss_functional = info_nce_loss(predictions, targets, temperature=1.0)
        
        loss_fn = InfoNCELoss(temperature=1.0)
        loss_module = loss_fn(predictions, targets)
        
        assert torch.allclose(loss_functional, loss_module), \
            "Functional and module interfaces should produce same result"


class TestInfoNCELossReduction:
    """Test reduction methods."""
    
    def test_mean_reduction(self):
        """Test mean reduction (default)."""
        predictions = torch.randn(8, 512, 1536)
        targets = torch.randn(8, 512, 1536)
        
        loss_fn = InfoNCELoss(reduction='mean')
        loss = loss_fn(predictions, targets)
        
        assert loss.ndim == 0
    
    def test_sum_reduction(self):
        """Test sum reduction."""
        predictions = torch.randn(8, 512, 1536)
        targets = torch.randn(8, 512, 1536)
        
        loss_fn_sum = InfoNCELoss(reduction='sum')
        loss_fn_mean = InfoNCELoss(reduction='mean')
        
        loss_sum = loss_fn_sum(predictions, targets)
        loss_mean = loss_fn_mean(predictions, targets)
        
        # Sum should be approximately batch_size * mean
        batch_size = 8
        assert torch.allclose(loss_sum, loss_mean * batch_size, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])