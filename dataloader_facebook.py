import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from collections import defaultdict
from datautil import load_rating_file_to_matrix, load_rating_file_to_list, load_negative_file, \
    load_group_member_to_dict, build_hyper_graph, build_group_graph
import logging

class FacebookGroupDataset(object):
    def __init__(self, data_dir, num_negatives=8, recent_k=200, group_profile_path=None):
        self.dataset_name = "facebook"
        self.data_dir = data_dir
        self.num_negatives = num_negatives
        self.recent_k = recent_k
        
        print(f"[{self.dataset_name.upper()}] loading from {data_dir}...")
        
        # Check if files exist
        required_files = [
            "userRatingTrain.txt", "userRatingTest.txt", "userRatingNegative.txt",
            "groupRatingTrain.txt", "groupRatingTest.txt", "groupRatingNegative.txt",
            "groupMember.txt"
        ]
        
        missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
        if missing:
            print(f"Missing files: {missing}. Please run preprocess_facebook.py first.")
            # Optionally call preprocess here if we import it
            # from preprocess_facebook import preprocess
            # preprocess(data_dir)
            raise FileNotFoundError("Run preprocess_facebook.py first!")

        # Load Mappings to get exact counts
        self.num_users = self._get_count(os.path.join(data_dir, "user_list.txt"))
        self.num_groups = self._get_count(os.path.join(data_dir, "group_list.txt"))
        self.num_items = self._get_count(os.path.join(data_dir, "item_list.txt"))
        
        print(f"Metadata: #Users {self.num_users}, #Groups {self.num_groups}, #Items {self.num_items}")

        # Paths
        user_path = os.path.join(data_dir, "userRating")
        group_path = os.path.join(data_dir, "groupRating")
        
        # User data
        # Note: load_rating_file_to_matrix expects filename without extension? 
        # No, in dataloader.py: load_rating_file_to_matrix(user_path + "Train.txt")
        # So here we pass prefix + "Train.txt"
        
        self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt", num_users=self.num_users, num_items=self.num_items)
        self.user_test_ratings = load_rating_file_to_list(user_path + "Test.txt")
        self.user_test_negatives = load_negative_file(user_path + "Negative.txt")
        
        print(f"UserItem: {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} interactions")

        # Group data
        self.group_train_matrix = load_rating_file_to_matrix(group_path + "Train.txt", num_users=self.num_groups, num_items=self.num_items)
        self.group_test_ratings = load_rating_file_to_list(group_path + "Test.txt")
        self.group_test_negatives = load_negative_file(group_path + "Negative.txt")
        self.group_member_dict = load_group_member_to_dict(os.path.join(data_dir, "groupMember.txt"))

        print(f"GroupItem: {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions")

        self.user_hist_mat = self._build_user_hist_mat(os.path.join(data_dir, "userRatingTrain.txt"))
        self.group_hist_ids, self.group_hist_mask = self._build_group_histories(os.path.join(data_dir, "groupRatingTrain.txt"), hist_len=self.recent_k)
        self.group_texts = self._load_group_texts(os.path.join(data_dir, "group_list.txt"), group_profile_path)

        # Member-level Hyper-graph
        # Note: build_hyper_graph expects group_train_path to read group-item interactions
        # It reads the file again.
        self.user_hyper_graph, self.item_hyper_graph, self.full_hg, group_data = build_hyper_graph(
            self.group_member_dict, group_path + "Train.txt", self.num_users, self.num_items, self.num_groups)
            
        # Group-level graph
        self.overlap_graph = build_group_graph(group_data, self.num_groups)
        
        print(f"\033[0;30;43m{self.dataset_name.upper()} finish loading!\033[0m")

    def _get_count(self, file_path):
        count = 0
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    count += 1
        return count

    def _load_id_item_sequences(self, file_path):
        seqs = defaultdict(list)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                obj_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                seqs[obj_id].extend(items)
        return seqs

    def _build_row_normalized_sparse(self, seqs, num_rows, num_cols, recent_k=None):
        indices = []
        values = []
        for row in range(num_rows):
            items = seqs.get(row, [])
            if recent_k is not None and recent_k > 0:
                items = items[-recent_k:]
            if not items:
                continue
            counts = defaultdict(int)
            for item in items:
                if 0 <= item < num_cols:
                    counts[item] += 1
            total = sum(counts.values())
            if total == 0:
                continue
            for item, c in counts.items():
                indices.append([row, item])
                values.append(float(c) / float(total))

        if not indices:
            empty_idx = torch.zeros((2, 0), dtype=torch.long)
            empty_val = torch.zeros((0,), dtype=torch.float32)
            return torch.sparse_coo_tensor(empty_idx, empty_val, size=(num_rows, num_cols))

        idx = torch.tensor(indices, dtype=torch.long).t().contiguous()
        val = torch.tensor(values, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, size=(num_rows, num_cols)).coalesce()

    def _build_user_hist_mat(self, user_train_file):
        seqs = self._load_id_item_sequences(user_train_file)
        return self._build_row_normalized_sparse(seqs, self.num_users, self.num_items, recent_k=None)

    def _build_group_histories(self, group_train_file, hist_len=200):
        seqs = self._load_id_item_sequences(group_train_file)
        hist_ids = torch.zeros((self.num_groups, hist_len), dtype=torch.long)
        hist_mask = torch.zeros((self.num_groups, hist_len), dtype=torch.float32)
        for gid in range(self.num_groups):
            items = seqs.get(gid, [])
            if hist_len > 0:
                items = items[-hist_len:]
            if not items:
                continue
            use_len = min(hist_len, len(items))
            hist_ids[gid, :use_len] = torch.tensor(items[:use_len], dtype=torch.long)
            hist_mask[gid, :use_len] = 1.0
        return hist_ids, hist_mask

    def _load_group_texts(self, group_list_path, group_profile_path=None):
        texts = [""] * self.num_groups
        if os.path.exists(group_list_path):
            with open(group_list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        gid = int(parts[0])
                        if 0 <= gid < self.num_groups:
                            texts[gid] = parts[1]

        if group_profile_path is not None and os.path.exists(group_profile_path):
            with open(group_profile_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 1:
                        continue
                    gid = int(parts[0])
                    name = parts[1] if len(parts) > 1 else ""
                    intro = parts[2] if len(parts) > 2 else ""
                    if 0 <= gid < self.num_groups:
                        merged = (name + " " + intro).strip()
                        if merged:
                            texts[gid] = merged

        return texts

    def get_train_instances(self, train):
        """Generate train samples (user, pos_item, neg_itm)"""
        users, pos_items, neg_items = [], [], []

        # train is a sparse matrix (dok)
        # keys are (u, i)
        
        keys = list(train.keys())
        # We need efficient lookup for negatives
        
        for (u, i) in keys:
            for _ in range(self.num_negatives):
                users.append(u)
                pos_items.append(i)

                j = np.random.randint(self.num_items)
                while (u, j) in train:
                    j = np.random.randint(self.num_items)
                neg_items.append(j)
        
        pos_neg_items = [[pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)]
        return users, pos_neg_items

    def get_user_dataloader(self, batch_size):
        users, pos_neg_items = self.get_train_instances(self.user_train_matrix)
        train_data = TensorDataset(torch.LongTensor(users), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_group_dataloader(self, batch_size):
        groups, pos_neg_items = self.get_train_instances(self.group_train_matrix)
        group_ids = torch.LongTensor(groups)
        hist = self.group_hist_ids[group_ids]
        mask = self.group_hist_mask[group_ids]
        train_data = TensorDataset(group_ids, torch.LongTensor(pos_neg_items), hist, mask)
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)
