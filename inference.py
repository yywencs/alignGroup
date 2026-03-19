import jieba
import jieba.posseg as pseg
import re
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import os
import json
from collections import defaultdict
from tqdm import tqdm
from model import AlignGroup
from dataloader_facebook import FacebookGroupDataset
from metrics import get_hit_k, get_ndcg_k

VALID_POS = {
    "n", "nr", "ns", "nt", "nz",
    "vn", "v", "vd", "vg", "vi", "vf", "vx",
    "a", "ad", "an", "ag", "al",
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

def generate_sentence_with_ollama(words, group_name):
    url = "http://localhost:11434/api/generate"
    model = "qwen3:32b"
    # prompt = f"这是一个群组常聊的十个词，请用这些关键词写一个通顺的句子表明群组经常在讨论什么，大概只有一半的词是准确的，可以过滤一些不对的词，要求句子自然、流畅，只要一句话：{', '.join(words)}"
    prompt = f"群组 {group_name} 常聊的十个词是：{', '.join(words)}。请用这些关键词写一个通顺的句子表明群组经常在讨论什么，大概只有一半的词是准确的，可以过滤一些不对的词，要求句子自然、流畅，只要一句话。"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # print(f"Calling Ollama ({model})...")
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                return result["response"].strip()
            else:
                return f"Error: No 'response' field in JSON. Keys: {list(result.keys())}"
        else:
            return f"Error: API returned status {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Exception calling Ollama: {str(e)}"

def load_id_map(file_path):
    id2name = {}
    name2id = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx = int(parts[0])
                name = parts[1]
                id2name[idx] = name
                name2id[name] = idx
    return id2name, name2id

def load_group_interactions(file_path):
    interactions = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                gid = int(parts[0])
                iid = int(parts[1])
                interactions[gid].append(iid)
    return interactions

def inference(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load Mappings
    data_dir = f"./data/{args.dataset}"
    item_map, item_name_map = load_id_map(os.path.join(data_dir, "item_list.txt"))
    group_map, _ = load_id_map(os.path.join(data_dir, "group_list.txt"))
    print(f"Loaded {len(item_map)} items and {len(group_map)} groups.")

    # 2. Load Dataset & Build Graphs
    # We use FacebookGroupDataset to handle graph construction
    print("Loading dataset and building graphs...")
    dataset = FacebookGroupDataset(
        data_dir=data_dir,
        num_negatives=1, # Not important for inference
        recent_k=args.recent_k,
        group_profile_path=args.group_profile_path
    )
    
    num_users = dataset.num_users
    num_items = dataset.num_items
    num_groups = dataset.num_groups

    user_hg = dataset.user_hyper_graph.to(device)
    item_hg = dataset.item_hyper_graph.to(device)
    full_hg = dataset.full_hg.to(device)
    try:
        overlap_graph = torch.Tensor(dataset.overlap_graph).to(device)
    except RuntimeError:
        overlap_graph = torch.Tensor(dataset.overlap_graph.tolist()).to(device)

    # 3. Load Model
    print("Initializing model...")
    # Load Item Embeddings if available
    item_embeddings = None
    emb_path = os.path.join(data_dir, 'item_embeddings.npy')
    if os.path.exists(emb_path):
        print(f"Loading pretrained item embeddings from {emb_path}")
        item_embeddings = np.load(emb_path)
        item_embeddings = torch.FloatTensor(item_embeddings).to(device)

    model = AlignGroup(
        num_users, num_items, num_groups, args,
        user_hg, item_hg, full_hg, overlap_graph,
        device, 
        cl_info=0.0, # Not used in inference
        temp=1.0,    # Not used in inference
        item_embeddings=item_embeddings,
        user_hist_mat=(dataset.user_hist_mat.to(device) if hasattr(dataset, "user_hist_mat") else None),
        group_texts=(dataset.group_texts if hasattr(dataset, "group_texts") else None),
        bge_model_path=args.bge_path
    )
    
    # Load Checkpoint
    checkpoint_path = os.path.join("checkpoints", args.checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    
    print("Precomputing static graph embeddings...")
    with torch.no_grad():
        current_item_emb = model.get_item_embedding()
        user_base = model._get_user_base_embedding(current_item_emb)
        group_emb_init = torch.zeros((num_groups, args.emb_dim), device=device)
        
        ui_emb, _ = model.hyper_graph_conv(
            user_base, current_item_emb, group_emb_init, num_users, num_items
        )
        _, i_emb_contextual = torch.split(ui_emb, [num_users, num_items])
        
    # Precompute all item embeddings for scoring
    # We use i_emb_contextual for scoring candidates
    candidate_embs = i_emb_contextual # Shape: [num_items, emb_dim]

    if args.json_input:
        print(f"Loading inference input from {args.json_input}")
        with open(args.json_input, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Parse JSON
        new_group_name = json_data.get("group_name", "Unknown Group")
        new_group_desc = json_data.get("group_description", "")
        chat_data = json_data.get("chat_data", [])
        
        print(f"Group: {new_group_name}")
        print(f"Description: {new_group_desc}")
        print(f"Chat data lines: {len(chat_data)}")
        
        # Load BGE Model locally to avoid reloading in loop
        print(f"Loading BGE model from {args.bge_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.bge_path)
        bge_model = AutoModel.from_pretrained(args.bge_path)
        bge_model.eval()
        bge_model.to(device)
        
        def get_bge_embeddings(texts):
            if not texts:
                return torch.tensor([], device=device)
            # Batching if necessary, but here texts usually small
            encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            with torch.no_grad():
                model_output = bge_model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0]
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings

        # 1. Compute Static Embedding from Name + Desc
        # Format: "Name Desc"
        group_text = f"{new_group_name} {new_group_desc}".strip()
        print(f"Encoding group text: {group_text}")
        
        # Encode using BGE
        bge_emb = get_bge_embeddings([group_text])
        
        # Project using model.group_text_projection
        if model.group_text_projection is not None:
            static_g = model.group_text_projection(bge_emb) # [1, emb_dim]
        else:
            print("Warning: model has no group_text_projection. Using zeros.")
            static_g = torch.zeros((1, args.emb_dim), device=device)
            
        # 2. Compute Dynamic Embedding from Chat Data
        valid_hist = []
        similarity_threshold = 0.7
        
        print(f"Processing {len(chat_data)} chat lines with POS tagging and BGE matching (thresh={similarity_threshold})...")
        
        for line in chat_data:
            line = line.strip()
            if not line:
                continue
            
            # Clean text
            content_clean = re.sub(r'http\S+', '', line)
            content_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content_clean)
            
            # POS Tagging
            words = list(pseg.cut(content_clean))
            candidates = []
            for w in words:
                token = w.word.strip()
                if len(token) < 2:
                    continue
                if not is_valid_pos(w.flag):
                    continue
                # Only consider words in our item map
                if token in item_name_map:
                    candidates.append(token)
            
            if not candidates:
                continue
            
            unique_candidates = list(set(candidates))
            
            # Compute embeddings
            cand_embs = get_bge_embeddings(unique_candidates)
            post_emb = get_bge_embeddings([line])
            
            # Similarity
            sims = torch.matmul(cand_embs, post_emb.t()).squeeze(1)
            
            # Filter
            for i, w in enumerate(unique_candidates):
                if sims[i].item() >= similarity_threshold:
                    valid_hist.append(item_name_map[w])
        
        print(f"Matched {len(valid_hist)} keywords in chat data.")
        
        if args.recent_k > 0 and len(valid_hist) > args.recent_k:
            valid_hist = valid_hist[-args.recent_k:]
            
        if not valid_hist:
            g_final = static_g
        else:
            length = len(valid_hist)
            hist_tensor = torch.zeros((1, args.recent_k), dtype=torch.long).to(device)
            mask_tensor = torch.zeros((1, args.recent_k), dtype=torch.float32).to(device)
            hist_tensor[0, :length] = torch.tensor(valid_hist, dtype=torch.long)
            mask_tensor[0, :length] = 1.0
            
            dynamic_g = model.dynamic_group_encoder(i_emb_contextual, hist_tensor, mask_tensor)
            
            if model.group_static_weight is not None:
                w = torch.sigmoid(model.group_static_weight)
                g_final = w * static_g + (1.0 - w) * dynamic_g
            else:
                g_final = dynamic_g
        
        # Score against all items
        if args.predictor == "MLP":
            g_expanded = g_final.expand(num_items, -1)
            combined = g_expanded * candidate_embs
            scores = model.predict(combined).squeeze(-1)
            scores = torch.sigmoid(scores)
        else:
            scores = torch.matmul(g_final, candidate_embs.t()).squeeze(0)
            
        # Mask history items?
        if valid_hist:
            scores[valid_hist] = -float('inf')
            
        topk_scores, topk_indices = torch.topk(scores, 10)
        topk_indices = topk_indices.cpu().numpy()
        
        top_words = [item_map.get(idx, f"Item_{idx}") for idx in topk_indices]
        
        print(f"\nConsensus Prediction for {new_group_name}:")
        print(f"Top-10 Consensus: {', '.join(top_words)}")
        
        print("\n--- Generating Sentence via LLM ---")
        sentence = generate_sentence_with_ollama(top_words, new_group_name)
        print(f"LLM Output:\n{sentence}")
        print("-" * 30)
        
        return

    # 4. Load Group History (Full Interactions)
    print("Loading group interactions history...")
    group_interactions = load_group_interactions(os.path.join(data_dir, "group_keyword_interactions.txt"))

    # 5. Load Test Cases
    test_file = os.path.join(data_dir, "groupRatingTest.txt")
    print(f"Loading test cases from {test_file}")
    test_cases = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                gid = int(parts[0])
                target_iid = int(parts[1])
                test_cases.append((gid, target_iid))
    
    print(f"Total test cases: {len(test_cases)}")
    
    # 5.1 Load Negative Samples for Metrics Calculation
    neg_file = os.path.join(data_dir, "groupRatingNegative.txt")
    print(f"Loading negative samples from {neg_file}")
    neg_samples = {}
    with open(neg_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: (gid,target_iid) neg1 neg2 ...
            try:
                key_part, neg_part = line.split(') ', 1)
                key_part = key_part.strip('(')
                gid_str, iid_str = key_part.split(',')
                gid, target_iid = int(gid_str), int(iid_str)
                negs = [int(x) for x in neg_part.split()]
                neg_samples[(gid, target_iid)] = negs
            except ValueError:
                continue

    results = []
    
    # Process each unique group in test set to avoid redundant computation if multiple targets exist for same group
    # But user wants per-line inference maybe? Let's group by GID first.
    test_groups = defaultdict(list)
    for gid, target in test_cases:
        test_groups[gid].append(target)
        
    print(f"Unique groups in test set: {len(test_groups)}")
    
    all_ranks = []
    k_list = [10]

    with torch.no_grad():
        for gid, targets in tqdm(test_groups.items(), desc="Inferring"):
            # Prepare History
            # Strategy: Use all interactions of this group, excluding the specific targets being tested?
            # Or just use the full history available in group_keyword_interactions.txt (which includes everything)
            # but mask out the targets?
            # For simplicity and "future prediction" simulation, let's take the history 
            # and remove ALL targets associated with this group in the test set.
            # This ensures no leakage.
            
            full_hist = group_interactions.get(gid, [])
            target_set = set(targets)
            valid_hist = [iid for iid in full_hist if iid not in target_set]
            
            # Truncate to recent_k
            if args.recent_k > 0:
                valid_hist = valid_hist[-args.recent_k:]
            
            # Compute Group Embedding
            group_inputs = torch.tensor([gid], dtype=torch.long).to(device)
            static_g = model._get_group_base_embedding(group_inputs)

            if not valid_hist:
                # Fallback if no history: use static embedding only (matching model.py logic)
                g_final = static_g
            else:
                length = len(valid_hist)
                hist_tensor = torch.zeros((1, args.recent_k), dtype=torch.long).to(device)
                mask_tensor = torch.zeros((1, args.recent_k), dtype=torch.float32).to(device)
                hist_tensor[0, :length] = torch.tensor(valid_hist, dtype=torch.long)
                mask_tensor[0, :length] = 1.0
            
                # Dynamic Part
                # dynamic_group_encoder expects (item_emb, history_ids, history_mask)
                dynamic_g = model.dynamic_group_encoder(i_emb_contextual, hist_tensor, mask_tensor)
                
                if model.group_static_weight is not None:
                    w = torch.sigmoid(model.group_static_weight)
                    g_final = w * static_g + (1.0 - w) * dynamic_g
                else:
                    g_final = dynamic_g
            
            # Score against all items
            # Predictor type
            if args.predictor == "MLP":
                # shape: [1, emb_dim] * [num_items, emb_dim] -> [num_items, emb_dim]
                # Then MLP -> [num_items, 1]
                # Optimization: Expand g_final
                g_expanded = g_final.expand(num_items, -1)
                combined = g_expanded * candidate_embs
                scores = model.predict(combined).squeeze(-1) # [num_items]
                scores = torch.sigmoid(scores)
            else:
                # Dot product
                scores = torch.matmul(g_final, candidate_embs.t()).squeeze(0) # [num_items]

            # Get Top 10
            # We should mask out items that are already in history? Usually yes for recommendation.
            # mask history items
            scores[valid_hist] = -float('inf')
            
            topk_scores, topk_indices = torch.topk(scores, 10)
            topk_indices = topk_indices.cpu().numpy()
            
            group_name = group_map.get(gid, f"Group_{gid}")
            
            # Print results for each target in this group
            for target in targets:
                target_word = item_map.get(target, f"Item_{target}")
                
                # Metrics Calculation logic
                # Need negative samples for this (gid, target) pair
                if (gid, target) in neg_samples:
                    negs = neg_samples[(gid, target)]
                    # Create candidate list: [target, neg1, neg2, ...]
                    eval_items = [target] + negs
                    eval_scores = scores[eval_items].cpu().numpy()
                    
                    # Rank: sort descending
                    # target is at index 0
                    # We want the rank of index 0
                    sorted_indices = np.argsort(-eval_scores)
                    rank = np.where(sorted_indices == 0)[0][0] # rank is 0-based index
                    
                    # Prepare pred_rank format for metrics.py: 
                    # metrics.py expects pred_rank where 0 is the target index.
                    # Since we only have one target per test case here, we can just append a row 
                    # where the value at the rank-th position is 0 (target) and others are > 0.
                    # Actually, get_hit_k takes pred_rank matrix where each row is the sorted indices of items?
                    # Let's check metrics.py:
                    # pred_rank = np.argsort(pred_score * -1, axis=1)
                    # hit = np.count_nonzero(pred_rank_k == 0) 
                    # This implies target index is 0 in the input score matrix.
                    # Yes, evaluate function puts target at index 0.
                    # So sorted_indices contains the indices of items in eval_items.
                    # If target (index 0) is in top k of sorted_indices, it's a hit.
                    
                    # Store sorted_indices for later batch metric calculation or calculate now
                    all_ranks.append(sorted_indices)
                
                if args.show_top10:
                    top_words = [item_map.get(idx, f"Item_{idx}") for idx in topk_indices]
                    
                    print(f"\nGroup: {group_name} (ID: {gid})")
                    print(f"Target Truth: {target_word}")
                    print(f"Top-10 Consensus: {', '.join(top_words)}")
    
    # Calculate overall metrics
    if all_ranks:
        all_ranks = np.array(all_ranks)
        print("\n" + "="*30)
        print("Evaluation Metrics:")
        print("="*30)
        for k in k_list:
            hit = get_hit_k(all_ranks, k)
            ndcg = get_ndcg_k(all_ranks, k)
            print(f"HR@{k}: {hit:.4f}")
            print(f"NDCG@{k}: {ndcg:.4f}")
        print("="*30)
    else:
        print("\nNo negative samples found for metrics calculation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="facebook_50")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--checkpoint", type=str, required=True, help="Filename of checkpoint in checkpoints/")
    parser.add_argument("--recent_k", type=int, default=50)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--group_profile_path", type=str, default="")
    parser.add_argument("--bge_path", type=str, default="/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/checkpoint/bge-large-zh")
    parser.add_argument("--json_input", type=str, help="Path to JSON file with group info for inference")
    parser.add_argument("--show_top10", action="store_true", help="Show Top-10 consensus prediction for each test case")
    
    # Dummy args for model init
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    inference(args)
