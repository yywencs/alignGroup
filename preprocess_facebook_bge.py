import os
import csv
import re
import random
import jieba
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import argparse

# Set random seed for reproducibility
random.seed(2025)

def preprocess(data_dir, model_path, similarity_threshold=0.5, limit=0):
    print("Processing Facebook dataset with BGE-based keyword extraction...")
    csv_path = os.path.join(data_dir, "data_merged.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load BGE Model
    print(f"Loading BGE model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Mappings
    user2id = {}
    group2id = {}
    item2id = {}
    
    # Interactions
    user_interactions = {} 
    group_interactions = {} 
    group_members = {} 

    # Stop words
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'it', 'that', 'this', 'from', 'as', 'be', 'by', 'are', 'was', 'were', 'will', 'have', 'has', 'had',
        '但', '和', '是', '的', '了', '在', '也', '很', '就', '都', '而', '及', '與', '着', '或', '一', '有', '我', '你', '他', '她', '它', '們',
        '个', '这', '那', '会', '去', '说', '想', '做', '人', '事', '时', '过', '着', '看', '到', '好', '多', '些', '等', '能', '会',
        'http', 'https', 'com', 'www', 'net', 'org', 'html', 'htm'
    ])

    # Read CSV
    raw_data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            raw_data.append(row)

    # Sort by date
    def parse_date(d_str):
        try:
            return d_str
        except:
            return "0000/00/00/00/00"
    
    raw_data.sort(key=lambda x: parse_date(x.get('Date Posted', '')))

    print(f"Total rows: {len(raw_data)}")
    
    if limit > 0:
        raw_data = raw_data[:limit]
        print(f"Limiting to first {limit} rows for testing.")

    # Cache for word embeddings to speed up processing
    word_embedding_cache = {}

    def get_embeddings(texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0]
                # normalize embeddings
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                all_embeddings.append(sentence_embeddings.cpu())
        if not all_embeddings:
            return torch.tensor([])
        return torch.cat(all_embeddings, dim=0)

    # Processing loop
    for row in tqdm(raw_data, desc="Processing Posts"):
        g_name = row['Group Name'].strip()
        u_name = row['User Name'].strip()
        content = row['Post Content'].strip()
        
        if not g_name or not u_name or not content:
            continue
            
        # Update IDs
        if g_name not in group2id:
            group2id[g_name] = len(group2id)
        gid = group2id[g_name]
        
        if u_name not in user2id:
            user2id[u_name] = len(user2id)
        uid = user2id[u_name]
        
        # Group Members
        if gid not in group_members:
            group_members[gid] = set()
        group_members[gid].add(uid)
        
        # Clean content
        content_clean = re.sub(r'http\S+', '', content)
        content_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content_clean)
        
        # Candidate words extraction
        words = list(jieba.cut(content_clean))
        candidates = []
        for w in words:
            w = w.strip()
            if len(w) < 2: continue
            if w.lower() in stop_words: continue
            candidates.append(w)
            
        if not candidates:
            continue
            
        # Deduplicate candidates for embedding calculation
        unique_candidates = list(set(candidates))
        
        # Check cache and identify new words
        new_words = [w for w in unique_candidates if w not in word_embedding_cache]
        
        # Compute embeddings for new words
        if new_words:
            new_embeddings = get_embeddings(new_words)
            for w, emb in zip(new_words, new_embeddings):
                word_embedding_cache[w] = emb
        
        # Get Post Embedding
        post_embedding = get_embeddings([content])[0] # Shape: [hidden_size]
        
        # Calculate similarity for each candidate
        # Stack candidate embeddings
        candidate_embeddings = torch.stack([word_embedding_cache[w] for w in unique_candidates]) # [num_candidates, hidden_size]
        
        # Compute Cosine Similarity
        # post_embedding is [hidden_size], candidate_embeddings is [num_candidates, hidden_size]
        # Both are normalized, so dot product is cosine similarity
        similarities = torch.matmul(candidate_embeddings, post_embedding.unsqueeze(1)).squeeze(1) # [num_candidates]
        
        # Filter
        valid_words = []
        for i, w in enumerate(unique_candidates):
            sim = similarities[i].item()
            if sim >= similarity_threshold:
                valid_words.append(w)
        
        # Add valid words to interactions
        for w in valid_words:
            if w not in item2id:
                item2id[w] = len(item2id)
            iid = item2id[w]
            
            # Add interactions
            if uid not in user_interactions:
                user_interactions[uid] = []
            user_interactions[uid].append(iid)
            
            if gid not in group_interactions:
                group_interactions[gid] = []
            group_interactions[gid].append(iid)

    print(f"Users: {len(user2id)}")
    print(f"Groups: {len(group2id)}")
    print(f"Items (Keywords): {len(item2id)}")

    # Calculate Item Frequencies
    item_counts = {}
    for obj_id, items in user_interactions.items():
        for iid in items:
            item_counts[iid] = item_counts.get(iid, 0) + 1
    for obj_id, items in group_interactions.items():
        for iid in items:
            item_counts[iid] = item_counts.get(iid, 0) + 1
            
    # Filter valid items for testing
    valid_test_items = set([iid for iid, c in item_counts.items() if c >= 5])
    print(f"Valid Test Items (Count >= 5): {len(valid_test_items)}")

    # Save Mappings
    with open(os.path.join(data_dir, 'group_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in group2id.items():
            f.write(f'{v} {k}\n')
            
    with open(os.path.join(data_dir, 'user_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in user2id.items():
            f.write(f'{v} {k}\n')
            
    with open(os.path.join(data_dir, 'item_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in item2id.items():
            f.write(f'{v} {k}\n')
            
    # Save Group Members
    with open(os.path.join(data_dir, 'groupMember.txt'), 'w', encoding='utf-8') as f:
        for gid in range(len(group2id)):
            if gid in group_members:
                members = group_members[gid]
                m_str = ','.join(map(str, members))
                f.write(f'{gid} {m_str}\n')
            else:
                f.write(f'{gid} \n')

    # Function to save interactions (Same as original)
    def save_interactions(interactions, prefix):
        print(f"Saving {prefix} interactions...")
        
        train_file = os.path.join(data_dir, f'{prefix}RatingTrain.txt')
        test_file = os.path.join(data_dir, f'{prefix}RatingTest.txt')
        neg_file = os.path.join(data_dir, f'{prefix}RatingNegative.txt')
        
        ftrain = open(train_file, 'w', encoding='utf-8')
        ftest = open(test_file, 'w', encoding='utf-8')
        fneg = open(neg_file, 'w', encoding='utf-8')
        
        all_items = list(item2id.values())
        
        for obj_id, items in interactions.items():
            # Deduplicate while preserving order
            items = list(dict.fromkeys(items)) 
            
            if len(items) < 4: 
                continue 
            
            k_test = 3
            test_candidates = []
            train_items = []
            
            idx = len(items) - 1
            while idx >= 0 and len(test_candidates) < k_test:
                curr_item = items[idx]
                if curr_item in valid_test_items:
                    test_candidates.append(curr_item)
                else:
                    train_items.append(curr_item)
                idx -= 1
            
            while idx >= 0:
                train_items.append(items[idx])
                idx -= 1
                
            train_items.reverse() # Restore order
            test_candidates.reverse() # Restore order
            
            if len(test_candidates) < k_test:
                 pass

            # Write Train
            if train_items:
                t_str = ' '.join(map(str, train_items))
                ftrain.write(f'{obj_id} {t_str}\n')
            
            # Write Test & Negative
            for target in test_candidates:
                negs = []
                while len(negs) < 100:
                    n = random.randint(0, len(item2id)-1)
                    if n not in items and n not in negs:
                        negs.append(n)
                
                neg_str = ' '.join(map(str, negs))
                # Test file: User Item
                ftest.write(f'{obj_id} {target}\n')
                # Negative file: (User, Item) Negs
                fneg.write(f'({obj_id},{target}) {neg_str}\n')

        ftrain.close()
        ftest.close()
        fneg.close()

    save_interactions(user_interactions, 'user')
    save_interactions(group_interactions, 'group')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Data directory')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of rows for testing')
    args = parser.parse_args()
    
    model_path = "/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/checkpoint/bge-large-zh"
    preprocess(args.data_dir, model_path, limit=args.limit)
