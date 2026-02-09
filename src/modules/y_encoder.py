"""
Y-encoder for VL-JEPA
"""
from typing import List
from huggingface_hub import GemmaTokenizer
import torch.nn as nn 

class YEncoder(nn.Module):
    """
    Y-Encoder of VL-JEPA
    """
    def __init__(self, model_name, max_length=512, output_dim=1536):
        super().__init__()
        self.tokenizer = GemmaTokenizer
        self.embedding_model = model_name
        self.max_length = max_length
        self.output_dim = output_dim
        self.projection = nn.Linear(self.embedding_model.hidden_dim, output_dim)
    
    def forward(self, text_list: List[str]):
        """
        Forward pass
        """
        pass