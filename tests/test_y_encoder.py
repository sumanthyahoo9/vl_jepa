"""
Unit tests for the Y-encoder
"""
import pytest
import torch
from src.modules.y_encoder import YEncoder

class TestYEncoderShapes:
    """
    Shape tests
    """
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_output_shape(self, batch_size):
        """
        Test the output shape
        """
        # Arrange
        text_samples = ["This is a test caption."] * batch_size
        # Initialize encoder
        encoder = YEncoder(
            model_name = "google/embeddinggemma-300m",
            max_length=512,
            output_dim=1536
        )
        # Act
        output = encoder(text_samples)
        # Assert
        expected_shape = (batch_size, 512, 1536)
        assert expected_shape == output.shape
    
    def test_single_sample(self):
        """Test with single text sample (B=1)."""
        text_samples = ["A dog running in the park."]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        output = encoder(text_samples)
        assert output.shape == (1, 512, 1536)

class TestYEncoderPadding:
    """Test padding behavior for different text lengths."""

    def test_short_text_padding(self):
        """Test that short text is padded to 512 tokens."""
        
        short_text = ["Hello"]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        output = encoder(short_text)
        assert output.shape == (1, 512, 1536)

    def test_long_text_truncation(self):
        """Test that long text is truncated to 512 tokens."""
        # Create very long text (likely > 512 tokens)
        long_text = [" ".join(["word"] * 1000)]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        output = encoder(long_text)
        assert output.shape == (1, 512, 1536)

    def test_varying_lengths_in_batch(self):
        """Test batch with varying text lengths."""
        text_samples = [
            "Short",
            "This is a medium length caption about something.",
            "This is a very long caption that goes on and on with many words to describe a scene in great detail."
        ]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        output = encoder(text_samples)
        assert output.shape == (3, 512, 1536)

class TestYEncoderTokenization:
    """Test tokenization behavior."""

    def test_token_ids_in_vocab_range(self):
        """Verify tokenized IDs are within valid vocabulary range."""
        
        text_samples = ["Test caption", "Another sample"]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        tokens = encoder.tokenize(text_samples)  # Should return [B, 512] of integers
        vocab_size = 262144  # From EmbeddingGemma config
        assert tokens['input_ids'].min() >= 0, f"Token IDs should be >= 0, got {tokens['input_ids'].min()}"
        assert tokens['input_ids'].max() < vocab_size, f"Token IDs should be < {vocab_size}, got {tokens['input_ids'].max()}"

    def test_special_tokens(self):
        """Verify special tokens (BOS, EOS, PAD) are used correctly."""
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        tokens = encoder.tokenize(["Short text"])
        assert tokens['input_ids'][0, 0] != 0, "First token should not be padding"

class TestYEncoderOutput:
    """Test output properties."""

    def test_output_dtype(self):
        """Verify output is float32."""
        text_samples = ["Test caption"]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        output = encoder(text_samples)
        assert output.dtype == torch.float32

    def test_no_nan_or_inf(self):
        """Verify output contains no NaN or Inf values."""
        text_samples = ["Normal caption", "Another one"]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        output = encoder(text_samples)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_deterministic_output(self):
        """Test that same input produces same output (in eval mode)."""
        text_samples = ["Deterministic test"]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        encoder.eval()
        with torch.no_grad():
            output1 = encoder(text_samples)
            output2 = encoder(text_samples)
        assert torch.allclose(output1, output2), "Outputs not deterministic"


class TestYEncoderProjection:
    """Test projection layer from 768 to 1536 dimensions."""

    def test_projection_layer_exists(self):
        """Verify projection layer properly transforms dimensions."""
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        # EmbeddingGemma outputs 768, should project to 1536
        assert hasattr(encoder, 'projection'), "Encoder should have projection layer"
        assert encoder.projection.in_features == 768
        assert encoder.projection.out_features == 1536


class TestYEncoderIntegration:
    """Integration tests for Y-Encoder."""

    def test_batch_processing(self):
        """Test processing multiple samples efficiently."""
        text_samples = [
            "First caption about a dog",
            "Second caption about a cat",
            "Third caption about a bird",
            "Fourth caption about a fish"
        ]
        encoder = YEncoder(model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536)
        output = encoder(text_samples)
        assert output.shape == (4, 512, 1536)
        # Each sample should produce different embeddings
        for i in range(len(text_samples) - 1):
            assert not torch.allclose(output[i], output[i+1]), f"Sample {i} and {i+1} produced identical embeddings"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
