import torch.nn as nn
import torch
from torch.quantization import quantize_dynamic
from bitsandbytes.nn import Linear4bit

class Connector(nn.Module):
    def __init__(self, config = None, quant: str = "4bit"):
        super().__init__()
        mlp_depth = 2
        modules = [nn.Linear(config.hidden_size*3, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp =  nn.Sequential(*modules) 
        self.quant = quant

        if self.quant == "8bit":
            target_dtype = torch.float32
        elif self.quant == "4bit":
            # rebuild with Linear4bit
            modules = [
                Linear4bit(
                    config.hidden_size * 3,      # input_features
                    config.hidden_size,          # output_features
                    True,                        # bias
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                ),
                nn.GELU(),
                Linear4bit(
                    config.hidden_size,      # input_features
                    config.hidden_size,          # output_features
                    True,                        # bias
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                )
            ]
            self.mlp = nn.Sequential(*modules)
            target_dtype = torch.bfloat16

        self.layernorm1 = nn.LayerNorm(config.hidden_size, dtype = target_dtype)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, dtype = target_dtype)
        self.alpha = nn.Parameter(torch.zeros(1, dtype = target_dtype))
        self.config = config
        self.target_dtype = target_dtype

    def forward(self, image_features, image_forward_outs, is_siglip=True):
        original_dtype = image_features.dtype

        dev = image_features.device
        image_features = image_features.to(device=dev, dtype=self.target_dtype)

        hidden = torch.stack(image_forward_outs.hidden_states, dim=0)

        image_features_1 = []
        image_features_2 = []
        if not is_siglip:
            g1 = hidden[0:12, :, 1:, :]
            g2 = hidden[12:24, :, 1:, :]
        else:
            g1 = hidden[0:13, :, :, :]
            g2 = hidden[13:26, :, :, :]

        image_features_1 = g1.mean(dim=0)
        image_features_2 = g2.mean(dim=0)

        # print(torch.cat([image_features, image_features_1, image_features_2], dim=-1).shape, self.config.hidden_size)
        fine_grained_image_features = self.mlp(torch.cat([image_features, image_features_1, image_features_2], dim=-1))

        img_feat_norm = self.layernorm1(image_features)
        fine_grained_norm = self.layernorm2(fine_grained_image_features)

        weight = torch.sigmoid(self.alpha)            # scalar in (0, 1)
        out = weight * fine_grained_norm + (1 - weight) * img_feat_norm

        return out.to(original_dtype)
        