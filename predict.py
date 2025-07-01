import os
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from model.connector import Connector
from model.builder import attach_connector_to_paligema
from transformers import BitsAndBytesConfig
import argparse
from utils import validate_paths, find_json_files
import json, tqdm
from PIL import Image
import re

def reload_from_trainer(ckpt_path: str, base_model_id: str, quant: str):
    processor = PaliGemmaProcessor.from_pretrained(base_model_id)

    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="cpu",
        quantization_config = quant_config,
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

    model.connector.load_state_dict(torch.load(f"{ckpt_path}/connector.pt"))
    model.multi_modal_projector.load_state_dict(f"{ckpt_path}/projector.pt")

    # 5) Wrap PEFT và load LoRA adapters
    model = PeftModel.from_pretrained(
        model,
        ckpt_path,
        is_trainable=False
    )
    model.to("cpu")

    return processor, model

def parse_args():
    parser = argparse.ArgumentParser(
        description="PaliGemma2 prediction script for medical image analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_id",
        type=str,
        default="google/paligemma2-10b-pt-224",
        help="HuggingFace model identifier for base PaliGemma2 model"
    )
    model_group.add_argument(
        "--ckpt",
        type=str,
        default="./paligemma2_ckpt/checkpoint-100",
        help="Path to LoRA checkpoint for fine-tuned weights"
    )

    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--base_dataset_path", 
        type=str, 
        default="original_dataset",
        help="Base path to dataset directory (e.g., original_dataset)"
    )
    data_group.add_argument(
        "--validation_type",
        type=str,
        choices=["hidden", "public", "test"],
        default="public",
        help="Type of validation dataset to use (hidden or public)"
    )
    data_group.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for prediction results"
    )
    data_group.add_argument(
        "--output_filename",
        type=str,
        default="predictions.json",
        help="Filename for the output predictions file"
    )

    inference_group = parser.add_argument_group('Inference Configuration')
    inference_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    inference_group.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on"
    )
    
    return parser.parse_args()

def run_predictions(args):
    """
    Main function to run predictions on all JSON files in the dataset directory.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Number of predictions made
    """
    # Construct full dataset path
    if args.validation_type == "test":
        dataset_path = os.path.join(args.base_dataset_path, f"test")
    else:
        dataset_path = os.path.join(args.base_dataset_path, f"validation-{args.validation_type}")
    
    # Validate paths and find JSON files
    print("Validating paths and discovering files...")
    input_files, checkpoint_exists = validate_paths(dataset_path, args.checkpoint_path)
    
    print(f"Found {len(input_files)} JSON files in {dataset_path}:")
    for file in input_files:
        print(f"  - {os.path.relpath(file, dataset_path)}")
    
    # Load model and processor
    print("\nLoading model and processor...")
    processor, model = reload_from_trainer(
        ckpt_path="fintuned_paligemma2_8bit_4_gpus/checkpoint-500",
        base_model_id="google/paligemma2-10b-pt-224",
        quant="8bit"
    )
    
    # Run predictions on all files
    print(f"\nRunning predictions...")
    all_predictions = []
    
    for input_file in input_files:
        predictions = predict_on_file(
            input_file, 
            model, 
            processor, 
            args.max_new_tokens,
            args.device
        )
        all_predictions.extend(predictions)
    
    # Save results
    print(f"\nSaving results...")
    output_dir = args.output_dir if args.output_dir else "."
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_filename)
    
    with open(output_file, "w") as f:
        json.dump(all_predictions, f, indent=2)
    

def predict_on_file(input_file, model, processor, max_new_tokens=1024, device="cuda:0"):
    """Perform predictions on a single JSON file containing questions and images."""
    IMAGE_TOKEN = "<image>"
    
    # Load data
    with open(input_file) as f:
        val_data = json.load(f)
    
    print(f"Processing {len(val_data)} samples from {os.path.basename(input_file)}")
    
    # Process each sample
    for sample in tqdm(val_data, desc=f"Predicting {os.path.basename(input_file)}"):
        try:
            # Handle image loading
            img_field = sample["ImageName"]
            if isinstance(img_field, list):
                img_paths = img_field[:5]  # Limit to 5 images max
            else:
                img_paths = [img_field]
            
            # Load and validate images
            imgs = []
            for img_path in img_paths:
                full_path = os.path.join(os.path.dirname(input_file), img_path)
                try:
                    img = Image.open(full_path).convert("RGB")
                    imgs.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    continue
            
            if not imgs:
                print(f"Warning: No valid images for sample, skipping")
                sample["Answer"] = "Error: No valid images"
                continue
            
            # Prepare input
            formatted_question = (
                "Analyze the given medical image and answer the following question:\n"
                f"Question: {sample['Question']}\n"
                "Please provide a clear and concise answer."
            )
            prefix = IMAGE_TOKEN * (processor.image_seq_length * len(imgs))
            input_text = f"{prefix}{processor.tokenizer.bos_token}{formatted_question}\n"

            # try 2 versions of prompt
            # input_text = f"{processor.tokenizer.bos_token}{prefix}{formatted_question}\n"
            
            # Process images and text
            pixel_values = processor.image_processor(imgs, return_tensors="pt")["pixel_values"].to(device)
            inputs = processor.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            input_len = inputs.input_ids.shape[-1]
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            # Decode output
            output = processor.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0][input_len:]
            
            # Parse answer based on task type
            parsed_answer = parse_answer(output, sample.get("TaskType", ""))
            sample["Answer"] = parsed_answer
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            sample["Answer"] = f"Error: {str(e)}"
    
    return val_data

def parse_answer(output, task_type=None):
    """Parse model output based on task type to extract the final answer."""
    output = output.strip()
    
    # Remove common prefixes
    if "Please provide a clear and concise answer." in output:
        try:
            output = output.split("Please provide a clear and concise answer.")[-1].strip()
        except:
            pass
    
    # Remove leading newlines
    if "\n" in output:
        output = output.split("\n", 1)[-1].strip()
    
    # Task-specific parsing
    task_type = (task_type or "").strip().lower()
    
    if task_type == "classification":
        return _parse_classification(output)
    elif task_type == "multi-label classification":
        return _parse_multi_label_classification(output)
    elif task_type in ["detection", "instance_detection"]:
        return _parse_detection(output)
    elif task_type in ["cell counting", "regression", "counting"]:
        return _parse_numeric(output)
    elif task_type == "report generation":
        return output
    else:
        return output


def _parse_classification(output):
    """Parse classification task output."""
    lines = output.splitlines()
    if len(lines) >= 1:
        last_line = lines[-1].strip()
        return last_line
    return output

def _parse_multi_label_classification(output):
    """Parse multi-label classification task output."""
    lines = output.splitlines()
    labels = []
    for line in lines:
        for part in re.split(r'[;]', line):
            label = part.strip()
            if label:
                labels.append(label)
    return "; ".join(labels)


def _parse_detection(output):
    """Parse detection task output (JSON format expected)."""
    match = re.search(r'\{.*\}|\[.*\]', output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return json.dumps(parsed)
        except:
            return match.group()
    return output


def _parse_numeric(output):
    """Parse numeric task output (counting, regression)."""
    match = re.search(r'[-+]?[0-9]*\.?[0-9]+', output)
    if match:
        return match.group()
    return "0"


if __name__ == "__main__":
    args = parse_args()

    proc, full_model = reload_from_trainer(
        ckpt_path=args.ckpt,
        base_model_id=args.model_id,
        quant="8bit"
    )
    full_model.eval()
    # Xác nhận
    # batch = next(iter(your_dataloader))  
    # batch = {k:v.to(full_model.device) for k,v in batch.items()}
    # out = full_model(**batch)
    # print("Reload OK, logits shape:", out.logits.shape)
