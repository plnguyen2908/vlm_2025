import os
import argparse
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig
from model.builder import attach_connector_to_paligema
from model.connector import Connector
from torch.utils.data import Dataset
from PIL import Image
import json, re, ast

# -----------------------------------------------------------------------------
# 1. Argument Parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/paligemma2-10b-pt-224")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="finetuned_pg2")
    parser.add_argument("--logging_dir", type=str, default="pg2_logs")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=10, 
        help="Number of steps between logging training metrics"
    )
    parser.add_argument(
        "--quant", choices=["4bit","8bit"], default="4bit",
        help="4bit for low vram, 8bit for more stable training"
    )
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# 2. Dataset & Collator
# -----------------------------------------------------------------------------
class ImageQADataset(Dataset):
    def __init__(self, ds, processor, max_images=5, image_token="<image>"):
        self.ds = ds
        self.processor = processor
        self.max_images = max_images
        self.image_token = image_token

        print(f"Dataset loaded with {len(self.ds)} samples")

    def __len__(self):
        return len(self.ds)

    def _boxes_to_tokens(self, boxes, image_size):
        """
        Convert a list of boxes into loc tokens.
        boxes: list of [x1, y1, x2, y2]
        image_size: (W, H)
        returns: list of <locXXXX> tokens for each coordinate in all boxes
        """
        W, H = image_size
        tokens = []
        for box in boxes:
            # each box is [x1, y1, x2, y2] (opencv's point style)
            box[0], box[1] = box[1], box[0]
            box[2], box[3] = box[3], box[2]
            # but paligemma is in matrix style
            for coord, dim in zip(box, [H, W, H, W]):
                idx = int((coord / dim) * 1024)
                idx = max(0, min(1023, idx))
                tokens.append(f"<loc{idx:04d}>")
        return tokens

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # print(sample)
        # prepare images list
        imgs = sample.get("image")
        imgs = imgs if isinstance(imgs, list) else [imgs]
        imgs = imgs[: self.max_images]
        pil_imgs = [Image.open(p).convert("RGB") for p in imgs]

        # formatted question
        num_image_tokens = self.processor.image_seq_length * len(pil_imgs)
        formatted_question = (
            f"{self.processor.tokenizer.bos_token}"
            f"{self.image_token * num_image_tokens}"
            "Analyze the given medical image and answer the following question:\n"
            f"Question: {sample.get('question', '')}\n"
            "Please provide a clear and concise answer."
        )

        raw_answer = sample.get("answer", "").strip()
        answer_text = raw_answer
        task_type = sample["task_type"]
        
        if task_type.lower() == "detection":
            # single-image detection
            W, H = pil_imgs[0].size
            bboxes = []
            if raw_answer.startswith("<loc"):
                tokens = re.findall(r"<loc\d{4}>", raw_answer)
                bboxes.append("".join(tokens))
            else:
                try:
                    boxes = ast.literal_eval(raw_answer)
                except Exception:
                    boxes = []
                for box in boxes:
                    tokens = self._boxes_to_tokens([box], (W, H))
                    bboxes.append("".join(tokens))
            answer_text = json.dumps(bboxes)

        elif task_type.lower() == "instance_detection":
            # multiple instances per category
            W, H = pil_imgs[0].size
            inst_map = {}
            if raw_answer.startswith("<loc"):
                # treat all tokens as one group
                tokens = re.findall(r"<loc\d{4}>", raw_answer)
                inst_map["0"] = ["".join(tokens)]
            else:
                try:
                    inst = ast.literal_eval(raw_answer)
                except Exception:
                    inst = {}
                for key, box_list in inst.items():
                    # box_list: list of [x1,y1,x2,y2]
                    # convert each box to continuous token string
                    token_strs = []
                    for box in box_list:
                        # pass list of one box to _boxes_to_tokens
                        box_tokens = self._boxes_to_tokens([box], (W, H))
                        token_strs.append("".join(box_tokens))
                    inst_map[str(key)] = token_strs
            # serialize grouping to JSON
            answer_text = json.dumps(inst_map)

            # print(answer_text)

        return {
            "images": pil_imgs,
            "question": formatted_question,
            "answer": answer_text,
            "task_type": task_type,
        }

def create_collate_fn(processor):
    def collate_fn(batch):
        imgs = sum((b["images"] for b in batch), [])
        with torch.no_grad():
            pv = processor.image_processor(imgs, return_tensors="pt")["pixel_values"].to(torch.bfloat16)
        questions = [sample["question"] for sample in batch]
        answers = [sample["answer"] for sample in batch]
        tokenized = processor.tokenizer(
            questions,
            text_pair=[answer + processor.tokenizer.eos_token for answer in answers],
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        input_ids = tokenized.input_ids.to(torch.long)
        attention_mask = tokenized.attention_mask.to(torch.long)
        token_type_ids = tokenized.token_type_ids.to(torch.long)
        # print(token_type_ids)
        labels = input_ids.masked_fill(token_type_ids == 0, -100).to(torch.long)

        return {
            "pixel_values": pv,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return collate_fn

# -----------------------------------------------------------------------------
# 3. Setup Configs: LoRA & Quantization
# -----------------------------------------------------------------------------
def setup_peft_and_quant(args):
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.quant=="4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    return lora_config, quant_config

# -----------------------------------------------------------------------------
# 4. Custom Trainer to save connector & projector
# -----------------------------------------------------------------------------
class CustomTrainer(Trainer):
    def __init__(self, processor, quant_config, base_model_id, lora_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.quant_config = quant_config
        self.base_model_id = base_model_id
        self.lora_args = lora_args

    def _save_checkpoint(self, model, trial, metrics=None):
        ckpt_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        out_dir = os.path.join(self.args.output_dir, ckpt_folder)

        # 1) Base save: config + LoRA adapter
        super()._save_checkpoint(model, trial)
        model.config.save_pretrained(out_dir)

        # 2) Save connector & projector
        if hasattr(model, 'connector'):
            torch.save(model.connector.state_dict(), os.path.join(out_dir, 'connector.pt'))
        if hasattr(model, 'multi_modal_projector'):
            torch.save(model.multi_modal_projector.state_dict(), os.path.join(out_dir, 'projector.pt'))

        # 3) Save base+connector without LoRA
        #    Instantiate new base and attach connector
        base_clone = PaliGemmaForConditionalGeneration.from_pretrained(
            self.base_model_id,
            quantization_config=self.quant_config,
            device_map='auto',
            low_cpu_mem_usage=True,
            attn_implementation='eager'
        )
        conn_clone = Connector(base_clone.config.vision_config, quant=self.lora_args.quant)
        base_clone = attach_connector_to_paligema(
            base_clone.config.vision_config, base_clone, connector=conn_clone
        )
        # Load matching params (no LoRA)
        base_clone.load_state_dict(model.state_dict(), strict=False)
        torch.save(base_clone.state_dict(), os.path.join(out_dir, 'pytorch_model.bin'))

        # 4) Save merged full model (base+connector+LoRA)
        # merged = model.merge_and_unload()
        # merged.save_pretrained(out_dir)

        # 5) Save quant config for inference
        if self.quant_config is not None:
            with open(os.path.join(out_dir, 'quant_config.json'), 'w') as f:
                json.dump(self.quant_config.to_dict(), f)

# -----------------------------------------------------------------------------
# 5. Main Training Pipeline
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    # model parallel:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # 4.1 Processor & Data
    proc = PaliGemmaProcessor.from_pretrained(args.model_id)
    train_ds = ImageQADataset(load_from_disk(args.train_data_path), proc)
    val_ds   = ImageQADataset(load_from_disk(args.val_data_path), proc)

    # 4.2 PEFT & Quantization
    lora_cfg, quant_cfg = setup_peft_and_quant(args)

    # 4.3 Load model với quant & device map

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant_cfg,
        # uncomment if you want model parallel instead of data parallel
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )


    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 4.4 Attach connector
    connector = Connector(model.config.vision_config, args.quant)
    model = attach_connector_to_paligema(model.config.vision_config, model, connector=connector)

    # 4.5 Apply LoRA
    model = get_peft_model(model, lora_cfg)

    # 4.6 Freeze all except LoRA & projector & connector
    for name, param in model.named_parameters():
        if not hasattr(param, 'dtype') or not param.dtype.is_floating_point:
            param.requires_grad = False
            continue
        if any(key in name for key in ['lora_', 'multi_modal_projector', 'connector']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.print_trainable_parameters()


    print("Modules with requires_grad=True and parameter counts:")

    # 4.7 TrainingArguments & Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,

        bf16=True, fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=1.0,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        eval_steps=args.eval_steps,
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,

        label_names=["labels"],
    )
    trainer = CustomTrainer(
        processor=proc,
        quant_config=quant_cfg,
        base_model_id=args.model_id,
        lora_args=args,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=create_collate_fn(proc)
    )

    # 4.8 Train
    trainer.train()
    trainer.save_model()

if __name__=="__main__":
    main()
