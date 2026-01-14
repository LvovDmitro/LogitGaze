import torch
import json
from pathlib import Path
import numpy as np
import torch.nn.functional as F

def create_logit_lens_json_last_layer(hidden_states, norm, lm_head, tokenizer, embedding_layer, model_name, image_filename, save_folder="/repo/gaze_heatmap_predi/llava-interp", top_k=5, image_size=336, patch_size=14, layer='last'):
    """
    Создает logit lens векторы для патчей изображения.
    
    Args:
        hidden_states: Hidden states от модели (список всех слоев)
        norm: Нормализация
        lm_head: Language model head
        tokenizer: Токенизатор
        embedding_layer: Слой эмбеддингов
        model_name: Имя модели
        image_filename: Путь к изображению
        save_folder: Папка для сохранения
        top_k: Количество топ токенов для каждого патча
        image_size: Размер изображения (по умолчанию 336 для LLaVA)
        patch_size: Размер патча (по умолчанию 14 для LLaVA)
        layer: Какой слой использовать: 'last', 'middle', или индекс слоя (int)
    """
    logits_folder = Path(save_folder) / "logits"
    semantics_folder = Path(save_folder) / "semantics"
    logits_folder.mkdir(parents=True, exist_ok=True)
    semantics_folder.mkdir(parents=True, exist_ok=True)

    # Выбираем нужный слой
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
    
    # Вычисляем количество патчей изображения
    # Для LLaVA: изображение 336x336 с патчами 14x14 = 24x24 = 576 патчей
    num_image_patches = (image_size // patch_size) ** 2  # 576 для 336x336
    
    # Находим индексы токенов патчей изображения
    # В LLaVA токены изображения идут после промпта "USER: <image>"
    # Нужно найти, где начинаются токены патчей
    # Обычно это после первых нескольких токенов промпта
    
    # Ищем токен <image> в промпте
    # Промпт: "USER: <image>\nFirst, summarize..."
    # После токенизации это будет примерно: [<s>, USER, :, <image>, <image>, ..., <image>, \n, First, ...]
    # Где <image> токены (ID 32000) повторяются num_image_patches раз
    
    # Простая эвристика: первые num_image_patches токенов после начала промпта - это патчи
    # Но нужно учесть, что промпт тоже токенизируется
    # Обычно структура: [<s>, USER, :, <image> x 576, \n, First, ...]
    # Значит патчи начинаются с индекса примерно 4-5
    
    # Более надежный способ: ищем токены с ID 32000 (<image>) в начале последовательности
    # Но в hidden_states уже обработанные токены, поэтому используем позиции
    
    # Для LLaVA-1.5 структура обычно такая:
    # - Токены 0-3: <s>, USER, :, пробел
    # - Токены 4-579: <image> токены (576 патчей)
    # - Токены 580+: текст промпта и ответа
    
    # Определяем начало токенов патчей изображения
    # В LLaVA структура обычно: [<s>, USER, :, пробел, <image>, <image>, ..., <image>, \n, First, ...]
    # Токены <image> имеют ID 32000 и повторяются num_image_patches раз
    
    # Более надежный способ: используем фиксированное смещение после промпта
    # Промпт "USER: <image>\n" токенизируется примерно как: [<s>, USER, :, пробел, <image> x 576, \n, ...]
    # Значит патчи начинаются с индекса 4 (после "<s>", "USER", ":", пробел)
    prompt_start_idx = 4  # После "<s>", "USER", ":", пробел
    image_patch_start_idx = prompt_start_idx
    image_patch_end_idx = image_patch_start_idx + num_image_patches
    
    # Проверяем, что у нас достаточно токенов
    if image_patch_end_idx > sequence_length:
        print(f"Warning: Sequence length ({sequence_length}) is less than expected patch end index ({image_patch_end_idx})")
        image_patch_end_idx = sequence_length
        num_image_patches = image_patch_end_idx - image_patch_start_idx
        print(f"Adjusting to {num_image_patches} patches")
    
    # Извлекаем векторы только для патчей изображения
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
    
    # Если не хватило токенов, дополняем нулями или повторяем последний
    if len(word_vectors) < num_image_patches:
        print(f"Warning: Found only {len(word_vectors)} image patch tokens, expected {num_image_patches}")
        # Дополняем последним вектором
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
    
    # Проверяем размерность: должно быть (num_image_patches, top_k, embedding_dim)
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
