import argparse
import torch
import numpy as np
import os
import logging
from collections import defaultdict
from model import AlignGroup
from dataloader_facebook import FacebookGroupDataset
from metrics import evaluate

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_base_group_name(group_name):
    # Split by __chunk and take the first part
    return group_name.split('__chunk')[0]

def inductive_split(dataset, groups_to_mask_names):
    """
    Split dataset into training and testing sets based on group names.
    Groups matching groups_to_mask_names will be used for testing (inductive).
    Others will be used for training.
    """
    
    # Identify IDs to mask
    test_group_ids = []
    train_group_ids = []
    
    # Read group list to map IDs to names
    group_map = {}
    with open(os.path.join(dataset.data_dir, "group_list.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                gid = int(parts[0])
                gname = parts[1]
                group_map[gid] = gname
                
                base_name = get_base_group_name(gname)
                if base_name in groups_to_mask_names:
                    test_group_ids.append(gid)
                else:
                    train_group_ids.append(gid)

    logging.info(f"Total Groups: {len(group_map)}")
    logging.info(f"Training Groups: {len(train_group_ids)}")
    logging.info(f"Testing (Inductive) Groups: {len(test_group_ids)}")
    logging.info(f"Masked Groups: {groups_to_mask_names}")
    
    return train_group_ids, test_group_ids, group_map

def train_inductive(args, dataset, train_group_ids, test_group_ids, group_map):
    running_device = torch.device(args.device)
    
    # Filter dataset for training
    # We need to construct a new dataloader that ONLY samples from train_group_ids
    # The original dataset.get_group_dataloader samples from all groups in group_train_matrix
    
    # We can create a custom subset of the training matrix or filter during iteration
    # For efficiency, let's filter the training keys
    
    train_keys = []
    # group_train_matrix is a dictionary (dok_matrix) or similar structure
    # keys are (group_id, item_id)
    
    logging.info("Filtering training data...")
    # Get all training instances first
    # This might be slow if we iterate all keys. 
    # dataset.group_train_matrix is scipy.sparse.dok_matrix
    
    # Faster way: iterate known train_group_ids and get their items?
    # But dok_matrix doesn't support fast row slicing easily if not csr.
    # Let's check dataloader implementation. load_rating_file_to_matrix returns dok_matrix.
    
    # To filter efficiently:
    train_group_set = set(train_group_ids)
    
    # We will subclass/monkey-patch the get_group_dataloader method to only return train groups
    original_get_train_instances = dataset.get_train_instances
    
    def get_inductive_train_instances(train_matrix):
        users, pos_items, neg_items = [], [], []
        keys = list(train_matrix.keys())
        
        for (u, i) in keys:
            if u in train_group_set:
                for _ in range(dataset.num_negatives):
                    users.append(u)
                    pos_items.append(i)
                    # Negative sampling
                    j = np.random.randint(dataset.num_items)
                    while (u, j) in train_matrix:
                        j = np.random.randint(dataset.num_items)
                    neg_items.append(j)
        
        pos_neg_items = [[pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)]
        return users, pos_neg_items

    # Prepare Model
    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    
    # Load Pretrained Item Embeddings if available
    item_embeddings = None
    emb_path = f'data/{args.dataset}/item_embeddings.npy'
    if os.path.exists(emb_path):
        item_embeddings = np.load(emb_path)
        item_embeddings = torch.FloatTensor(item_embeddings).to(running_device)
        
    # Graph structures need to be on device
    user_hg = dataset.user_hyper_graph.to(running_device)
    item_hg = dataset.item_hyper_graph.to(running_device)
    full_hg = dataset.full_hg.to(running_device)
    try:
        overlap_graph = torch.Tensor(dataset.overlap_graph).to(running_device)
    except RuntimeError:
        overlap_graph = torch.Tensor(dataset.overlap_graph.tolist()).to(running_device)

    model = AlignGroup(num_users, num_items, num_groups, args, user_hg, item_hg,
                       full_hg, overlap_graph, running_device, args.cl_weight, args.temp,
                       item_embeddings=item_embeddings,
                       user_hist_mat=dataset.user_hist_mat.to(running_device),
                       group_texts=dataset.group_texts,
                       bge_model_path=args.bge_path)
    model.to(running_device)
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    
    logging.info("Starting Inductive Training...")
    
    for epoch in range(args.epoch):
        model.train()
        
        # 1. Group Training Loop
        groups, pos_neg_items = get_inductive_train_instances(dataset.group_train_matrix)
        
        batch_size = args.batch_size
        num_batches = (len(groups) + batch_size - 1) // batch_size
        if num_batches == 0:
            num_batches = 1
            
        epoch_group_loss = 0.0
        
        # Shuffle
        indices = np.arange(len(groups))
        np.random.shuffle(indices)
        
        from tqdm import tqdm
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epoch} [Group]")
        
        for b_idx in pbar:
            batch_indices = indices[b_idx * batch_size : (b_idx + 1) * batch_size]
            
            batch_groups = [groups[i] for i in batch_indices]
            batch_pos_neg = [pos_neg_items[i] for i in batch_indices]
            
            u = torch.LongTensor(batch_groups).to(running_device)
            pi_ni = torch.LongTensor(batch_pos_neg).to(running_device)
            
            # Get History
            # dataset.group_hist_ids is a CPU tensor, u is a CUDA tensor
            # We need to use CPU indices to index CPU tensor, then move result to GPU
            g_hist = dataset.group_hist_ids[u.cpu()].to(running_device)
            g_mask = dataset.group_hist_mask[u.cpu()].to(running_device)
            
            # Get Members
            members = [torch.LongTensor(dataset.group_member_dict[gid]).to(running_device) for gid in batch_groups]
            
            optimizer.zero_grad()
            
            pos_items_input = pi_ni[:, 0]
            neg_items_input = pi_ni[:, 1]
            
            loss, _ = model(u, None, pos_items_input, neg_items_input, members, 'train',
                            group_history=g_hist, group_mask=g_mask)
            
            loss.backward()
            optimizer.step()
            epoch_group_loss += loss.item()
            pbar.set_postfix({'g_loss': f"{epoch_group_loss/(b_idx+1):.4f}"})
            
        logging.info(f"Epoch {epoch+1}/{args.epoch} Group Loss: {epoch_group_loss/num_batches:.4f}")

        # 2. User Training Loop
        user_dataloader = dataset.get_user_dataloader(batch_size)
        epoch_user_loss = 0.0
        user_batches = 0
        
        pbar_user = tqdm(user_dataloader, desc=f"Epoch {epoch+1}/{args.epoch} [User]")
        
        for batch_data in pbar_user:
            batch_users, batch_pos_neg = batch_data
            
            u = torch.LongTensor(batch_users).to(running_device)
            pi_ni = torch.LongTensor(batch_pos_neg).to(running_device)
            
            optimizer.zero_grad()
            
            pos_items_input = pi_ni[:, 0]
            neg_items_input = pi_ni[:, 1]
            
            loss, _ = model(u, None, pos_items_input, neg_items_input, [], 'user')
            
            loss.backward()
            optimizer.step()
            epoch_user_loss += loss.item()
            user_batches += 1
            pbar_user.set_postfix({'u_loss': f"{epoch_user_loss/user_batches:.4f}"})

        logging.info(f"Epoch {epoch+1}/{args.epoch} User Loss: {epoch_user_loss/max(1, user_batches):.4f}")

    return model

def test_inductive(model, dataset, test_group_ids, args):
    logging.info("\n=== Testing Inductive (Cold Start) Performance ===")
    model.eval()
    running_device = torch.device(args.device)
    
    hits = []
    ndcgs = []
    
    # Prepare test data for unseen groups
    # We use dataset.group_test_ratings but only for test_group_ids
    
    test_group_set = set(test_group_ids)
    
    test_ratings = []
    test_negatives = []
    test_gids = []
    
    for idx, (gid, item_id) in enumerate(dataset.group_test_ratings):
        if gid in test_group_set:
            test_ratings.append([gid, item_id])
            test_negatives.append(dataset.group_test_negatives[idx])
            test_gids.append(gid)
            
    if not test_ratings:
        logging.warning("No test ratings found for masked groups!")
        return

    logging.info(f"Testing on {len(test_ratings)} interactions from unseen groups...")

    # For inductive testing, we MUST NOT use the Group ID embedding.
    # We must use predict_cold_start logic.
    # However, predict_cold_start returns TopK indices.
    # Here we need to score specific (Positive + Negatives) items to calculate Hit/NDCG.
    
    # Let's adapt the evaluation logic to use cold_start style inference
    
    with torch.no_grad():
        # Pre-compute Static and Dynamic Embeddings for these groups manually
        # to simulate "Cold Start" (No ID lookup)
        
        # 1. Get Group Texts and History
        batch_texts = [dataset.group_texts[gid] for gid in test_gids]
        # test_gids is a list of ints, dataset.group_hist_ids is CPU tensor
        batch_hist_ids = dataset.group_hist_ids[test_gids].to(running_device)
        batch_hist_mask = dataset.group_hist_mask[test_gids].to(running_device)
        
        # 2. Encode Text (BGE)
        # Note: In a real script, batching this is better. Here we do all at once or small batches.
        # Assuming memory fits for small test set.
        bge_emb = model._encode_texts_bge(batch_texts, model.bge_model_path, running_device)
        static_g = model.group_text_projection(bge_emb)
        
        # 3. Encode History
        # We need updated Item Embeddings from the trained model
        current_item_emb = model.get_item_embedding()
        
        # Refine item embeddings via HyperGraph (as in predict_cold_start)
        user_base = model._get_user_base_embedding(current_item_emb)
        group_emb_init = torch.zeros((dataset.num_groups, model.emb_dim), device=running_device) # Dummy init
        ui_emb, _ = model.hyper_graph_conv(user_base, current_item_emb, group_emb_init, dataset.num_users, dataset.num_items)
        _, refined_i_emb = torch.split(ui_emb, [dataset.num_users, dataset.num_items])
        
        dynamic_g = model.dynamic_group_encoder(refined_i_emb, batch_hist_ids, batch_hist_mask)
        
        # 4. Combine
        if model.group_static_weight is not None:
            w = torch.sigmoid(model.group_static_weight)
            g_use = w * static_g + (1.0 - w) * dynamic_g
        else:
            g_use = dynamic_g
            
        # 5. Calculate Scores for Test Items (Pos + Negs)
        # test_ratings: [[gid, pos_item], ...]
        # test_negatives: [[neg1, neg2...], ...]
        
        for i in range(len(test_ratings)):
            gid = test_ratings[i][0]
            pos_item = test_ratings[i][1]
            neg_items = test_negatives[i]
            
            candidates = [pos_item] + neg_items
            cand_tensor = torch.LongTensor(candidates).to(running_device)
            
            # Group Vector
            g_vec = g_use[i] # (emb_dim)
            
            # Item Vectors
            i_vecs = refined_i_emb[cand_tensor] # (101, emb_dim)
            
            # Score
            scores = torch.matmul(i_vecs, g_vec)
            
            # Rank
            scores_np = scores.cpu().numpy()
            # Sort descending
            rank_indices = np.argsort(-scores_np)
            
            # Ground truth is at index 0
            # Find where index 0 is in rank_indices
            rank = np.where(rank_indices == 0)[0][0]
            
            # Metrics
            # Hit@K
            for k in args.topK:
                if rank < k:
                    hits.append(1.0)
                else:
                    hits.append(0.0)
                    
            # NDCG@K
            # ndcg = log(2) / log(rank + 2) if rank < k
            # Note: The provided metrics.py calculates NDCG slightly differently, let's match standard or provided
            # Standard: IDCG is 1 (since 1 pos item). DCG = 1 / log2(rank+2)
            
    # Aggregate Metrics
    # Since we appended per-k per-instance, we need to restructure
    # Actually hits is flat list? No, let's use separate lists
    
    # Redo metric collection
    hit_results = {k: [] for k in args.topK}
    ndcg_results = {k: [] for k in args.topK}
    
    import math
    
    for i in range(len(test_ratings)):
        # ... (same scoring logic)
        gid = test_ratings[i][0]
        pos_item = test_ratings[i][1]
        neg_items = test_negatives[i]
        
        candidates = [pos_item] + neg_items
        cand_tensor = torch.LongTensor(candidates).to(running_device)
        
        g_vec = g_use[i]
        i_vecs = refined_i_emb[cand_tensor]
        scores = torch.matmul(i_vecs, g_vec)
        
        scores_np = scores.cpu().numpy()
        rank_indices = np.argsort(-scores_np)
        rank = np.where(rank_indices == 0)[0][0]
        
        for k in args.topK:
            if rank < k:
                hit_results[k].append(1.0)
                ndcg_results[k].append(math.log(2) / math.log(rank + 2))
            else:
                hit_results[k].append(0.0)
                ndcg_results[k].append(0.0)

    for k in args.topK:
        avg_hit = np.mean(hit_results[k])
        avg_ndcg = np.mean(ndcg_results[k])
        logging.info(f"Inductive Hit@{k}: {avg_hit:.5f}, NDCG@{k}: {avg_ndcg:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="facebook")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=20) # Fewer epochs for quick test
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--recent_k", type=int, default=200)
    parser.add_argument("--bge_path", type=str, default="BAAI/bge-m3") # Use default or arg
    parser.add_argument("--cl_weight", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--topK", type=list, default=[1, 5, 10])
    
    args = parser.parse_args()
    
    setup_logging()
    
    # 1. Load Dataset
    dataset = FacebookGroupDataset(data_dir=f"./data/{args.dataset}", num_negatives=args.num_negatives, recent_k=args.recent_k)
    
    # 2. Define Groups to Mask (Inductive Test Set)
    # Selected based on group_list.txt content
    # "台灣機器學習與人工智慧同好會", "靠北職場大小事（靠北老闆、靠北員工、靠北同事…一起來靠北）", "桃園在地生活大小事"
    groups_to_mask = [
        "台灣機器學習與人工智慧同好會", 
        "靠北職場大小事（靠北老闆、靠北員工、靠北同事…一起來靠北）",
        "桃園在地生活大小事"
    ]
    
    # 3. Split
    train_gids, test_gids, group_map = inductive_split(dataset, groups_to_mask)
    
    if not test_gids:
        logging.error("No groups matched the mask list! Check exact names.")
        exit()
        
    # 4. Train with Masking
    model = train_inductive(args, dataset, train_gids, test_gids, group_map)
    
    # 5. Test Inductively
    test_inductive(model, dataset, test_gids, args)
