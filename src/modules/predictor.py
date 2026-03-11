"""
Predictor module
"""
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel

class Predictor(nn.Module):
    """
    Predictor for VL-JEPA.
    
    Combines vision and text embeddings, processes through Llama layers,
    predicts target text embeddings.
    
    Architecture:
    - Input projections: 1536 → 2048
    - Llama 3.2-1B (first 8 layers frozen, last 8 trainable)
    - Output projection: 2048 → 1536
    """
    
    def __init__(
        self,
        model_name: str = "facebook/layerskip-llama3.2-1B",
        input_dim: int = 1536,
        hidden_dim: int = 2048,
        output_dim: int = 1536
    ):
        super().__init__()
        
        # Load full Llama model
        print(f"Loading {model_name}...")
        self.llama_model = AutoModel.from_pretrained(model_name)
        
        # Freeze first 8 layers (only last 8 are trainable)
        for i in range(8):
            for param in self.llama_model.layers[i].parameters():
                param.requires_grad = False
        
        # Projection layers
        self.vision_proj = nn.Linear(input_dim, hidden_dim)
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Convert to bfloat16 to match Llama
        self.to(torch.bfloat16)
        
        print(f"Predictor initialized: {self.get_num_params()/1e6:.1f}M trainable params")
    
    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, Sv: torch.Tensor, Xq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Sv: Vision embeddings [B, T*256, 1536]
            Xq: Text query embeddings [B, 512, 1536]
        
        Returns:
            Predicted text embeddings [B, 512, 1536]
        """
        B, _, text_seq_len = Sv.shape[0], Sv.shape[1], Xq.shape[1]
        
        # Project inputs: 1536 → 2048
        vision_projected = self.vision_proj(Sv)
        text_projected = self.text_proj(Xq)
        
        # Concatenate: vision first, text last
        hidden_states = torch.cat([vision_projected, text_projected], dim=1)
        
        # Process through Llama (handles position embeddings internally)
        output = self.llama_model(inputs_embeds=hidden_states)
        hidden_states = output.last_hidden_state
        
        # Extract last 512 positions (text query positions)
        text_hidden = hidden_states[:, -text_seq_len:, :]
        
        # Project output: 2048 → 1536
        predicted = self.output_proj(text_hidden)
        
        return predicted

if __name__ == "__main__":
    print("Testing the Predictor")
    batch_size = 2
    num_frames=4
    vision_seq_len = num_frames*256 # T * 256 = 1024
    text_seq_len = 512
    Sv = torch.randn(batch_size, vision_seq_len, 1536).bfloat16()
    Xq = torch.randn(batch_size, text_seq_len, 1536).bfloat16()
    
    # Initialize predictor
    predictor = Predictor()
    predictor.eval()
    with torch.no_grad():
        output = predictor(Sv, Xq)

    print("\nInput shapes:")
    print(f"  Vision: {Sv.shape}")
    print(f"  Text query: {Xq.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {text_seq_len}, 1536]")
    
    assert output.shape == (batch_size, text_seq_len, 1536), \
        f"Shape mismatch: {output.shape}"
    
    print("\n✓ Shape test passed!")
    print(f"✓ Predictor has {predictor.get_num_params()/1e6:.1f}M parameters")

