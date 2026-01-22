import os
import csv
import re
import random
import sys
from collections import Counter

import jieba
import jieba.posseg as pseg
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import argparse

random.seed(2025)

VALID_POS = {
    "n",
    "nr",
    "ns",
    "nt",
    "nz",
    "vn",
    "v",
    "vd",
    "vg",
    "vi",
    "vf",
    "vx",
    "a",
    "ad",
    "an",
    "ag",
    "al",
    "eng",
}


def is_valid_pos(flag):
    if not flag:
        return False
    if flag in VALID_POS:
        return True
    if flag[0] in {"n", "v", "a"}:
        return True
    return False

def extract_keywords_from_text(model_path, text, similarity_threshold=0.5, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def get_embeddings(texts):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0]
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                all_embeddings.append(sentence_embeddings.cpu())
        if not all_embeddings:
            return torch.tensor([])
        return torch.cat(all_embeddings, dim=0)

    content_clean = re.sub(r'http\S+', '', text)
    content_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content_clean)

    words = list(pseg.cut(content_clean))
    candidates = []
    for w in words:
        token = w.word.strip()
        if len(token) < 2:
            continue
        if not is_valid_pos(w.flag):
            continue
        candidates.append(token)

    if not candidates:
        return {
            "text": text,
            "content_clean": content_clean,
            "jieba_words": words,
            "candidates": [],
            "unique_candidates": [],
            "selected": [],
            "scored": [],
        }

    unique_candidates = list(set(candidates))
    candidate_embeddings = get_embeddings(unique_candidates)
    post_embedding = get_embeddings([text])[0]
    similarities = torch.matmul(candidate_embeddings, post_embedding.unsqueeze(1)).squeeze(1)

    scored = []
    selected = []
    for i, w in enumerate(unique_candidates):
        sim = float(similarities[i].item())
        scored.append((w, sim))
        if sim >= similarity_threshold:
            selected.append(w)

    scored.sort(key=lambda x: x[1], reverse=True)
    selected_scored = [(w, sim) for (w, sim) in scored if sim >= similarity_threshold]

    return {
        "text": text,
        "content_clean": content_clean,
        "jieba_words": words,
        "candidates": candidates,
        "unique_candidates": unique_candidates,
        "selected": [w for (w, _) in selected_scored],
        "scored": scored,
    }

def demo_bge_output(model_path, text):
    print(f"Loading BGE model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    embedding = sentence_embeddings[0].detach().cpu().numpy()
    print("text:", text)
    print("embedding_shape:", tuple(embedding.shape))
    print("embedding_norm:", float(np.linalg.norm(embedding)))
    print("embedding_head16:", embedding[:16].tolist())
    print("embedding_tail4:", embedding[-4:].tolist() if embedding.shape[0] >= 4 else embedding.tolist())

def preprocess(
    data_dir,
    model_path,
    similarity_threshold=0.5,
    limit=0,
    virtual_group_window=50,
    max_vocab_size=5000,
):
    print("Processing Facebook dataset with BGE-based keyword extraction...")
    csv_path = os.path.join("/home/yangyw/code/my_code/rz/AlignGroup/data", "data_merged.csv")
    
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
    
    # Interactions metadata
    group_members = {}
    item_counts = Counter()

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

    print("Building POS-filtered vocabulary with global frequency statistics...")
    vocab_counter = Counter()
    for row in tqdm(raw_data, desc="Building vocab"):
        content = row.get("Post Content", "")
        content = content.strip()
        if not content:
            continue
        content_clean = re.sub(r'http\S+', '', content)
        content_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content_clean)
        for w in pseg.cut(content_clean):
            token = w.word.strip()
            if len(token) < 2:
                continue
            if not is_valid_pos(w.flag):
                continue
            vocab_counter[token] += 1

    if max_vocab_size is not None and max_vocab_size > 0:
        most_common = vocab_counter.most_common(max_vocab_size)
    else:
        most_common = list(vocab_counter.items())
    vocab_tokens = [w for w, _ in most_common]
    vocab_set = set(vocab_tokens)
    print(f"Vocab size after POS filtering and top-{max_vocab_size}: {len(vocab_tokens)}")

    word_embedding_cache = {}
    group_post_counts = {}

    user_hist_path = os.path.join(data_dir, 'user_keyword_interactions.txt')
    group_hist_path = os.path.join(data_dir, 'group_keyword_interactions.txt')

    def get_embeddings(texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0]
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                all_embeddings.append(sentence_embeddings.cpu())
        if not all_embeddings:
            return torch.tensor([])
        return torch.cat(all_embeddings, dim=0)

    with open(user_hist_path, 'w', encoding='utf-8') as user_hist_file, \
         open(group_hist_path, 'w', encoding='utf-8') as group_hist_file:
        for row in tqdm(raw_data, desc="Processing Posts"):
            orig_g_name = row['Group Name'].strip()
            u_name = row['User Name'].strip()
            content = row['Post Content'].strip()
            
            if not orig_g_name or not u_name or not content:
                continue

            prev_count = group_post_counts.get(orig_g_name, 0)
            chunk_idx = prev_count // max(1, virtual_group_window)
            group_post_counts[orig_g_name] = prev_count + 1
            virtual_name = f"{orig_g_name}__chunk{chunk_idx}"

            if virtual_name not in group2id:
                group2id[virtual_name] = len(group2id)
            gid = group2id[virtual_name]
            
            if u_name not in user2id:
                user2id[u_name] = len(user2id)
            uid = user2id[u_name]
            
            if gid not in group_members:
                group_members[gid] = set()
            group_members[gid].add(uid)
            
            content_clean = re.sub(r'http\S+', '', content)
            content_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content_clean)
            
            words = list(pseg.cut(content_clean))
            candidates = []
            for w in words:
                token = w.word.strip()
                if len(token) < 2:
                    continue
                if not is_valid_pos(w.flag):
                    continue
                if vocab_set and token not in vocab_set:
                    continue
                candidates.append(token)
                
            if not candidates:
                continue
                
            unique_candidates = list(set(candidates))
            
            new_words = [w for w in unique_candidates if w not in word_embedding_cache]
            
            if new_words:
                new_embeddings = get_embeddings(new_words)
                for w, emb in zip(new_words, new_embeddings):
                    word_embedding_cache[w] = emb
            
            post_embedding = get_embeddings([content])[0]
            
            candidate_embeddings = torch.stack([word_embedding_cache[w] for w in unique_candidates])
            
            similarities = torch.matmul(candidate_embeddings, post_embedding.unsqueeze(1)).squeeze(1)
            
            valid_words = []
            for i, w in enumerate(unique_candidates):
                sim = similarities[i].item()
                if sim >= similarity_threshold:
                    valid_words.append(w)
            
            for w in valid_words:
                if w not in item2id:
                    item2id[w] = len(item2id)
                iid = item2id[w]

                item_counts[iid] += 2

                user_hist_file.write(f"{uid} {iid}\n")
                group_hist_file.write(f"{gid} {iid}\n")

    print(f"Users: {len(user2id)}")
    print(f"Groups: {len(group2id)}")
    print(f"Items (Keywords): {len(item2id)}")

    def load_interactions(path):
        interactions = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                obj_id = int(parts[0])
                iid = int(parts[1])
                if obj_id not in interactions:
                    interactions[obj_id] = []
                interactions[obj_id].append(iid)
        return interactions

    user_interactions = load_interactions(user_hist_path)
    group_interactions = load_interactions(group_hist_path)

    id2item = {v: k for k, v in item2id.items()}
    id2group = {v: k for k, v in group2id.items()}
    from collections import Counter as C2
    print("Sample keywords for each group:")
    for gid in range(len(group2id)):
        items = group_interactions.get(gid, [])
        if not items:
            continue
        freq = C2(items)
        top_words = []
        for iid, c in freq.most_common(20):
            token = id2item.get(iid, "")
            if token:
                top_words.append(f"{token}({c})")
        group_name = id2group.get(gid, "")
        print(f"GID {gid}\t{group_name}")
        print("  keywords:", ", ".join(top_words))

    valid_test_items = set([iid for iid, c in item_counts.items() if c >= 5])
    print(f"Valid Test Items (Count >= 5): {len(valid_test_items)}")

    with open(os.path.join(data_dir, 'group_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in group2id.items():
            f.write(f'{v} {k}\n')
            
    with open(os.path.join(data_dir, 'user_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in user2id.items():
            f.write(f'{v} {k}\n')
            
    with open(os.path.join(data_dir, 'item_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in item2id.items():
            f.write(f'{v} {k}\n')
            
    with open(os.path.join(data_dir, 'groupMember.txt'), 'w', encoding='utf-8') as f:
        for gid in range(len(group2id)):
            if gid in group_members:
                members = group_members[gid]
                m_str = ','.join(map(str, members))
                f.write(f'{gid} {m_str}\n')
            else:
                f.write(f'{gid} \n')

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
            for ti in train_items:
                ftrain.write(f'{obj_id} {ti} 1\n')
            
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
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    parser.add_argument('--virtual_group_window', type=int, default=50)
    parser.add_argument('--max_vocab_size', type=int, default=5000)
    parser.add_argument('--limit', type=int, default=0, help='Limit number of rows for testing')
    args = parser.parse_args()
    
    model_path = "/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/checkpoint/bge-large-zh"
    preprocess(
        args.data_dir,
        model_path,
        similarity_threshold=args.similarity_threshold,
        limit=args.limit,
        virtual_group_window=args.virtual_group_window,
        max_vocab_size=args.max_vocab_size,
    )
