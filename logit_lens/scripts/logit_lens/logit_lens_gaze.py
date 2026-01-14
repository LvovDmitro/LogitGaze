import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import torch

# Add project root to sys.path so that src/ modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.HookedLVLM import HookedLVLM
from src.lvlm_lens_gze import create_logit_lens_json_last_layer

def is_image_file(filename):
    """Check if a file is a valid image."""
    valid_extensions = ('.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG')
    return filename.lower().endswith(valid_extensions)

def get_image_paths(root_folder):
    """Recursively retrieve all image file paths from the root folder and its subdirectories."""
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if is_image_file(filename):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def process_images(image_folder, save_folder, device, quantize_type, num_images=None):
    """Process images in a folder and generate logit lens outputs."""
    if not os.path.isdir(image_folder):
        raise ValueError(f"{image_folder} is not a valid directory.")

    model = HookedLVLM(device=device, quantize=True, quantize_type=quantize_type)
    # In your process_images function, replace lines 35-36 with:

# Get the correct base model depending on the architecture
    language_model = model.model.language_model

    # For Llama-based models (like LLaVA uses), the structure might be:
    if hasattr(language_model, 'model'):
        # This handles cases like LlamaModel where there's a nested .model
        base_model = language_model.model
    elif hasattr(language_model, 'layers') or hasattr(language_model, 'decoder'):
        # Direct access for models that don't have nested .model
        base_model = language_model
    else:
        base_model = language_model

    # Now access norm and lm_head correctly
    if hasattr(base_model, 'norm'):
        norm = base_model.norm
    elif hasattr(language_model, 'norm'):  # Fallback
        norm = language_model.norm
    else:
        # Handle different normalization layer names across architectures
        if hasattr(base_model, 'final_layernorm'):  # For some models
            norm = base_model.final_layernorm
        elif hasattr(base_model, 'ln_f'):  # For GPT-style models
            norm = base_model.ln_f
        else:
            raise AttributeError(f"Could not find normalization layer in model: {type(base_model)}")

    # For lm_head, try different possible locations
    if hasattr(model.model, 'lm_head'):
        lm_head = model.model.lm_head
    elif hasattr(language_model, 'lm_head'):
        lm_head = language_model.lm_head
    else:
        raise AttributeError("Could not find lm_head in model. Tried model.model.lm_head and language_model.lm_head")
    
    tokenizer = model.processor.tokenizer
    model_name = model.model.config._name_or_path.split("/")[-1]
    embedding_layer = torch.nn.Embedding.from_pretrained(lm_head.weight)

    image_paths = get_image_paths(image_folder)
    if not image_paths:
        raise ValueError(f"No valid image files found in {image_folder}.")

    if num_images:
        image_paths = image_paths[:num_images]

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not open {image_path}. Error: {e}")
            continue

        text_question = "First, summarize the image in one sentence. " \
        "Then, describe each small part of image in detail with objects, textures, and colors."
        prompt = f"USER: <image>\n{text_question} ASSISTANT:"
        hidden_states = model.forward(image, prompt, output_hidden_states=True).hidden_states

        create_logit_lens_json_last_layer(
            hidden_states, norm, lm_head, tokenizer, embedding_layer, model_name, image_path, save_folder,
            top_k=5, image_size=336, patch_size=14, layer='last'
        )

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process images using HookedLVLM model")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images")
    parser.add_argument("--save_folder", required=True, help="Path to save the results")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument("--quantize_type", default="fp16", help="Quantization type")
    parser.add_argument("--num_images", type=int, help="Number of images to process (optional)")

    args = parser.parse_args()

    process_images(
        image_folder=args.image_folder,
        save_folder=args.save_folder,
        device=args.device,
        quantize_type=args.quantize_type,
        num_images=args.num_images
    )

if __name__ == "__main__":
    main()