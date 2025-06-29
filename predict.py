import os
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from model.connector import Connector
from model.builder import attach_connector_to_paligema

def reload_from_trainer(ckpt_path: str, base_model_id: str, quant: str):
    processor = PaliGemmaProcessor.from_pretrained(base_model_id)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    # 3) Tạo và attach connector
    connector = Connector(model.config.vision_config, quant=quant)
    model = attach_connector_to_paligema(
        model.config.vision_config,
        model,
        connector=connector
    )

    # 5) Wrap PEFT và load LoRA adapters
    model = PeftModel.from_pretrained(
        model,
        ckpt_path,
        is_trainable=False
    )

    return processor, model

if __name__ == "__main__":
    proc, full_model = reload_from_trainer(
        ckpt_path="fintuned_paligemma2_8bit_4_gpus/checkpoint-500",
        base_model_id="google/paligemma2-10b-pt-224",
        quant="8bit"
    )
    full_model.eval()
    # Xác nhận
    # batch = next(iter(your_dataloader))  
    # batch = {k:v.to(full_model.device) for k,v in batch.items()}
    # out = full_model(**batch)
    # print("Reload OK, logits shape:", out.logits.shape)
