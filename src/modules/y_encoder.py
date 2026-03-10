"""
Y-encoder for VL-JEPA
"""
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn 

class YEncoder(nn.Module):
    """
    Y-Encoder of VL-JEPA
    """
    def __init__(self, model_name="google/embeddinggemma-300m", max_length=512, output_dim=1536):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
        self.embedding_model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.projection = nn.Linear(768, output_dim)
    
    def tokenize(self, text_list: List[str]) -> dict:
        """
        Tokenize the text samples with truncated length
        """
        tokens = self.tokenizer(
            text_list,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return tokens
    
    def forward(self, text_list: List[str]):
        """
        Forward pass of VL-JEPA
        """
        # Tokenize text
        tokens = self.tokenize(text_list)
        
        # Move tokens to same device as model
        device = next(self.embedding_model.parameters()).device
        tokens = self.tokenize(text_list)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output_projections = self.embedding_model(tokens["input_ids"], attention_mask=tokens['attention_mask'])
        embeddings = output_projections.last_hidden_state
        embeddings = self.projection(embeddings)
        return embeddings
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick test
    encoder = YEncoder()
    print(f"Y-Encoder initialized with {encoder.get_num_params():,} parameters")
    
    # Test with sample text
    sample_texts = [
        "A dog running in the park",
        "A cat sitting on a couch",
        "Birds flying in the sky"
    ]
    
    encoder.eval()
    with torch.no_grad():
        output = encoder(sample_texts)
    
    print(f"Input: {len(sample_texts)} text samples")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [3, 512, 1536]")
    assert output.shape == (3, 512, 1536), f"Shape mismatch: {output.shape}"
    print("✓ Shape test passed!")
