import os
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from model.connector import Connector
from model.builder import attach_connector_to_paligema

def reload_from_trainer(output_dir: str, base_model_id: str, quant: str):
    # 1) Processor
    processor = PaliGemmaProcessor.from_pretrained(base_model_id)

    # 2) Base model với cùng quant config
    #    (load_in_4bit/8bit được lưu trong pytorch_model.bin, HF tự nhận)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 3) Tạo và attach connector
    connector = Connector(model.config.vision_config, quant=quant)
    model = attach_connector_to_paligema(
        model.config.vision_config,
        model,
        connector=connector
    )

    # 4) Load full state_dict (base + projector + connector + LoRA)
    state = torch.load(os.path.join(output_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state, strict=False)

    # 5) Wrap PEFT và load LoRA adapters
    model = PeftModel.from_pretrained(
        model,
        output_dir,
        is_trainable=False
    )

    return processor, model

if __name__ == "__main__":
    proc, full_model = reload_from_trainer(
        output_dir="fintuned_pg2",
        base_model_id="google/paligemma2-10b-pt-224",
        quant="4bit"
    )
    full_model.eval()
    # Xác nhận
    batch = next(iter(your_dataloader))  
    batch = {k:v.to(full_model.device) for k,v in batch.items()}
    out = full_model(**batch)
    print("Reload OK, logits shape:", out.logits.shape)
