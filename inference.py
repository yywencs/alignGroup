import jieba
import jieba.posseg as pseg
import re
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

    # 6. Inference Loop
    # To speed up, we can batch process, but simple loop is fine for demonstration
    # We need to compute 'i_emb' (Contextual Item Embedding) from HyperGraphConv first.
    # Since HyperGraphConv depends on user_emb, item_emb, group_emb_init (zeros), 
    # and these don't change per query, we can precompute them once.
    
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

    results = []
    
    # Process each unique group in test set to avoid redundant computation if multiple targets exist for same group
    # But user wants per-line inference maybe? Let's group by GID first.
    test_groups = defaultdict(list)
    for gid, target in test_cases:
        test_groups[gid].append(target)
        
    print(f"Unique groups in test set: {len(test_groups)}")

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
                
                top_words = [item_map.get(idx, f"Item_{idx}") for idx in topk_indices]
                
                print(f"\nGroup: {group_name} (ID: {gid})")
                print(f"Target Truth: {target_word}")
                print(f"Top-10 Consensus: {', '.join(top_words)}")
                if target in topk_indices:
                    print(f"Result: HIT")
                else:
                    print(f"Result: MISS")

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
    
    # Dummy args for model init
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    inference(args)
