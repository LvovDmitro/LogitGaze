import torch
import json
from pathlib import Path
import numpy as np
import torch.nn.functional as F

def create_logit_lens_json_last_layer(
    hidden_states,
    norm,
    lm_head,
    tokenizer,
    embedding_layer,
    model_name,
    image_filename,
    save_folder="./data/logit_lens_TP",
    top_k=5,
    image_size=336,
    patch_size=14,
    layer="last",
):
    """
    Create logit lens vectors for image patches and save them to disk.

    Args:
        hidden_states: Hidden states from the model (list for all layers)
        norm: Normalization layer
        lm_head: Language model head
        tokenizer: Tokenizer
        embedding_layer: Embedding layer
        model_name: Model name
        image_filename: Path to the input image
        save_folder: Root folder to save outputs (will contain 'logits/' and 'semantics/')
        top_k: Number of top tokens per image patch
        image_size: Image size (default 336 for LLaVA)
        patch_size: Patch size (default 14 for LLaVA)
        layer: Which layer to use: 'last', 'middle', or explicit integer index
    """
    logits_folder = Path(save_folder) / "logits"
    semantics_folder = Path(save_folder) / "semantics"
    logits_folder.mkdir(parents=True, exist_ok=True)
    semantics_folder.mkdir(parents=True, exist_ok=True)

    # Select the requested layer
    if layer == 'last':
        selected_layer_hidden_states = hidden_states[-1]
        layer_idx = len(hidden_states) - 1
    elif layer == 'middle':
        layer_idx = len(hidden_states) // 2
        selected_layer_hidden_states = hidden_states[layer_idx]
    elif isinstance(layer, int):
        layer_idx = layer
        selected_layer_hidden_states = hidden_states[layer_idx]
    else:
        raise ValueError(f"Invalid layer parameter: {layer}. Use 'last', 'middle', or int index")
    
    last_layer_hidden_states = selected_layer_hidden_states  
    normalized = norm(last_layer_hidden_states)
    logits = lm_head(normalized)
    probs = torch.softmax(logits, dim=-1)

    sequence_length = last_layer_hidden_states.size(1)
    
    # Compute number of image patches
    # For LLaVA: image 336x336 with 14x14 patches → 24x24 = 576 patches
    num_image_patches = (image_size // patch_size) ** 2  # 576 for 336x336
    
    # Locate token indices corresponding to image patches.
    # In LLaVA tokens for the image come after the prompt "USER: <image>".
    #
    # A typical tokenized structure:
    #   [<s>, USER, :, <image>, <image>, ..., <image>, \\n, First, ...]
    # where <image> tokens (ID 32000) repeat num_image_patches times.
    #
    # Heuristic layout for LLaVA-1.5:
    #   tokens 0–3   : <s>, USER, :, space
    #   tokens 4–579 : <image> tokens (576 patches)
    #   tokens 580+  : text of the prompt and answer
    #
    # In practice we use a fixed offset after the text prompt:
    #   \"USER: <image>\\n\" → roughly [<s>, USER, :, space, <image> x 576, \\n, ...]
    # so patches start from index 4.
    prompt_start_idx = 4  # After \"<s>\", \"USER\", \":\", space
    image_patch_start_idx = prompt_start_idx
    image_patch_end_idx = image_patch_start_idx + num_image_patches
    
    # Check that we have enough tokens
    if image_patch_end_idx > sequence_length:
        print(f"Warning: Sequence length ({sequence_length}) is less than expected patch end index ({image_patch_end_idx})")
        image_patch_end_idx = sequence_length
        num_image_patches = image_patch_end_idx - image_patch_start_idx
        print(f"Adjusting to {num_image_patches} patches")
    
    # Extract vectors only for image patches
    word_vectors = []  
    last_layer_top_tokens = []

    for pos in range(image_patch_start_idx, min(image_patch_end_idx, sequence_length)):
        top_values, top_indices = torch.topk(probs[0, pos], k=top_k)

        token_ids = top_indices.tolist()
        token_embeddings = embedding_layer(torch.tensor(token_ids, device=embedding_layer.weight.device))
        
        word_vectors.append(token_embeddings.detach().cpu().numpy())

        top_words = [tokenizer.convert_ids_to_tokens(idx) for idx in token_ids]
        last_layer_top_tokens.append({
            "position": pos,
            "patch_index": pos - image_patch_start_idx,
            "tokens": top_words,
            "probs": [float(prob.item()) for prob in top_values]
        })
    
    # If we do not have enough tokens, pad with the last vector
    if len(word_vectors) < num_image_patches:
        print(f"Warning: Found only {len(word_vectors)} image patch tokens, expected {num_image_patches}")
        # Pad using the last available vector
        if len(word_vectors) > 0:
            last_vector = word_vectors[-1]
            while len(word_vectors) < num_image_patches:
                word_vectors.append(last_vector.copy())
                patch_idx = len(word_vectors) - 1
                last_layer_top_tokens.append({
                    "position": image_patch_start_idx + patch_idx,
                    "patch_index": patch_idx,
                    "tokens": last_layer_top_tokens[-1]["tokens"],
                    "probs": last_layer_top_tokens[-1]["probs"]
                })

    word_vectors_array = np.stack(word_vectors, axis=0)
    
    # Sanity check: shape must be (num_image_patches, top_k, embedding_dim)
    assert word_vectors_array.shape[0] == num_image_patches, \
        f"Expected {num_image_patches} patches, got {word_vectors_array.shape[0]}"

    npy_filename = f"{Path(image_filename).stem}_word_vectors.npy"
    npy_path = semantics_folder / npy_filename
    np.save(npy_path, word_vectors_array)

    # Debugging: Verify if word vectors similar top words
    # for idx, (word_vector, token_word) in enumerate(zip(word_vectors, last_layer_top_tokens)):
    #     decoded_word = embedding_to_word(torch.tensor(word_vector).mean(dim=0), embedding_layer, tokenizer)

    #     print(f"Position {idx}:")
    #     print(f"  - Extracted top word: {token_word['tokens'][0]}")
    #     print(f"  - Word from embedding vector: {decoded_word}")
    #     print("-" * 50)

    # Create JSON metadata
    json_data = {
        "image_filename": image_filename,
        "word_vectors_file": str(npy_path.name),
        "num_patches": num_image_patches,
        "patch_size": patch_size,
        "image_size": image_size,
        "patch_start_idx": image_patch_start_idx,
        "patch_end_idx": image_patch_end_idx,
        "word_vectors_shape": list(word_vectors_array.shape),
        "layer_used": layer_idx,
        "total_layers": len(hidden_states),
        "last_layer_data": last_layer_top_tokens
    }

    output_filename = f"{model_name}_{Path(image_filename).stem}_word_vectors.json"
    output_path = logits_folder / output_filename
    with open(output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

def embedding_to_word(embedding_vector, embedding_layer, tokenizer):
    """
    Convert an embedding vector back into a word using cosine similarity.
    """
    device = embedding_layer.weight.device
    embedding_vector = embedding_vector.to(device)

    all_token_embeddings = embedding_layer.weight  
    similarity = F.cosine_similarity(embedding_vector.unsqueeze(0), all_token_embeddings, dim=-1)
    closest_token_id = torch.argmax(similarity).item()

    return tokenizer.convert_ids_to_tokens(closest_token_id)
