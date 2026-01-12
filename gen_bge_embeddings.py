import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import sys

def generate_embeddings(data_dir, model_path, batch_size=128):
    print(f"Loading BGE model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Model loaded on {device}.")

    # Load items
    item_list_path = os.path.join(data_dir, 'item_list.txt')
    if not os.path.exists(item_list_path):
        print(f"{item_list_path} not found.")
        return

    items = []
    ids = []
    
    print("Reading item list...")
    with open(item_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 2:
                ids.append(int(parts[0]))
                items.append(parts[1])
            elif len(parts) == 1:
                ids.append(int(parts[0]))
                items.append("") # Empty content
    
    if not ids:
        print("No items found.")
        return

    max_id = max(ids)
    num_items = max_id + 1
    
    ordered_texts = [""] * num_items
    for idx, text in zip(ids, items):
        if idx < num_items:
            ordered_texts[idx] = text
        
    print(f"Total items: {num_items}")
    
    all_embeddings = []
    
    print("Generating embeddings...")
    with torch.no_grad():
        for i in range(0, num_items, batch_size):
            batch_texts = ordered_texts[i:i+batch_size]
            
            # Tokenize
            # Use 'Query: ' prefix? BGE usually expects 'Represent this sentence for searching relevant passages: ' for queries.
            # But here items are 'passages' or just semantic units.
            # For general semantic similarity, raw text is usually fine or "passage: " prefix.
            # BGE 1.5 doesn't strictly require instruction for passages.
            # Let's use raw text.
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # Use CLS token
            embeddings = outputs.last_hidden_state[:, 0]
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {i}/{num_items}")
                
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    save_path = os.path.join(data_dir, 'item_embeddings.npy')
    np.save(save_path, all_embeddings)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    data_dir = "data/facebook"
    model_path = "/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/checkpoint/bge-large-zh"
    generate_embeddings(data_dir, model_path)
