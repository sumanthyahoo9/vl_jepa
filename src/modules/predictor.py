"""
Predictor module
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class Predictor(nn.Module):
    """
    Predictor that projects vision and text together 
    """
    def __init__(self, model_name="facebook/layerskip-llama3.2-1B"):
        super().__init__()
        self.vision_projection = nn.Linear(1536, 2048)
        self.text_projection = nn.Linear(1536, 2048)
        full_model = AutoModel.from_pretrained(model_name)
        self.llama_layers = AutoModel("")
        self.output_proj = nn.Linear(2048, 1536)
    
    def forward(self, text_input, vision_input):
        """
        Forward pass
        """
        # 1. Project inputs
        vision_projected = self.vision_projection(vision_input)
        text_projected = self.text_projection(text_input)
        # 2. Concatenate
        concatenated_input = torch.concat([vision_projected, text_projected], dim=1)
        # 3. Process through Llama
        llama_output = self.llama_layers(concatenated_input, attention_mask=False)
        # 4. Extract last 512
        text_outputs = llama_output[:,-512:,:]
        # 5. Project output
        output_projections = self.output_proj(text_outputs)
        return output_projections
