import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .connector import Connector 

def attach_connector_to_paligema(
    config,
    paligema: PreTrainedModel, 
    connector: nn.Module = None
) -> PreTrainedModel:
    """
    Given a loaded PaliGemmaForConditionalGeneration (or any HF VLM
    with .vision_tower and .multi_modal_projector), monkey-patch its
    get_image_features method so that it runs `connector` on the raw
    hidden_states before projecting.
    
    Usage:
        from transformers import AutoModelForConditionalGeneration
        from connector_wrapper import attach_connector_to_paligema
        from connectors import MyConnector

        model = AutoModelForConditionalGeneration.from_pretrained("…/paligemma")
        connector = MyConnector(…)
        model = attach_connector_to_paligema(model, connector)
    """
    # default to identity connector if none provided
    if connector is None:
        connector = Connector(config)

    # The actual vision→projector lives in model.model (PaliGemmaModel)
    pmodel = paligema.model
    pmodel.add_module("connector", connector)
    # paligema.add_module("connector", connector)

    def new_get_image_features(pixel_values: torch.Tensor) -> torch.Tensor:
        # 1) run vision_tower with hidden_states
        outputs = pmodel.vision_tower(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = outputs.hidden_states[-1]     # (layers, B, L, D)
        
        # move to correct gpu because you divide a model to multiple gpus
        device = hs.device
        pmodel.connector.to(device)

        # 2) run your external connector\
        if getattr(pmodel.connector, "quant", None) == "8bit":
            inp = hs.float()                # dynamic-int8 connector just need float32 input
        else:
            inp = hs 

        feats_fp32 = pmodel.connector(inp, outputs, True)

        # cast back to bfloat16 before projecting
        feats = feats_fp32.to(hs.dtype)  

        # 3) feed into the pretained projector
        proj = pmodel.multi_modal_projector(feats) 
        # 4) keep original scaling
        scaled = proj / (paligema.config.text_config.hidden_size ** 0.5)
        return scaled.to(hs.dtype)
    
    # overwrite both model.model and convenience wrapper on paligema
    pmodel.get_image_features = new_get_image_features
    # paligema.get_image_features = new_get_image_features

    return paligema
