import os
import csv
import re
import random
import jieba
import sys

# Set random seed for reproducibility
random.seed(2025)

def preprocess(data_dir):
    print("Processing Facebook dataset...")
    csv_path = os.path.join(data_dir, "data_merged.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Mappings
    user2id = {}
    group2id = {}
    item2id = {}
    
    # Interactions
    # user_interactions[uid] = [iid1, iid2, ...] (ordered by time ideally, but CSV has time)
    user_interactions = {} 
    group_interactions = {} # gid -> [iid1, iid2, ...]
    group_members = {} # gid -> set(uid)

    # Stop words (basic list, can be expanded)
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'it', 'that', 'this', 'from', 'as', 'be', 'by', 'are', 'was', 'were', 'will', 'have', 'has', 'had',
        '但', '和', '是', '的', '了', '在', '也', '很', '就', '都', '而', '及', '與', '着', '或', '一', '有', '我', '你', '他', '她', '它', '們',
        '个', '这', '那', '会', '去', '说', '想', '做', '人', '事', '时', '过', '着', '看', '到', '好', '多', '些', '等', '能', '会',
        'http', 'https', 'com', 'www', 'net', 'org', 'html', 'htm'
    ])

    # Read CSV
    # Expected columns: Group Name, User Name, Post Content, ...
    # We should sort by date if possible to have temporal split, but random split is also fine for now.
    # The CSV has 'Date Posted' (e.g., 2026/01/04/11/46). We can try to sort.
    
    raw_data = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        # Normalize field names
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            raw_data.append(row)
            
    # Sort by date
    # Date format example: 2026/01/04/11/46 (YYYY/MM/DD/HH/MM)
    def parse_date(d_str):
        try:
            return d_str
        except:
            return "0000/00/00/00/00"
    
    raw_data.sort(key=lambda x: parse_date(x.get('Date Posted', '')))

    print(f"Total rows: {len(raw_data)}")

    for row in raw_data:
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
        
        # Tokenize content
        # Remove URLs
        content = re.sub(r'http\S+', '', content)
        # Remove special chars
        content = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content)
        
        words = jieba.cut(content)
        
        for w in words:
            w = w.strip()
            if len(w) < 2: continue # Skip single chars? Maybe keep for Chinese? Let's skip very short stuff.
            if w.lower() in stop_words: continue
            
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
    # Calculate Item Frequencies
    item_counts = {}
    for obj_id, items in user_interactions.items():
        for iid in items:
            item_counts[iid] = item_counts.get(iid, 0) + 1
    for obj_id, items in group_interactions.items():
        for iid in items:
            item_counts[iid] = item_counts.get(iid, 0) + 1

    print(f"Items (Keywords): {len(item2id)}")
    
    # Filter valid items for testing
    # Items with count >= 5 are valid for GT
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

    # Function to save interactions
    def save_interactions(interactions, prefix):
        print(f"Saving {prefix} interactions...")
        
        train_file = os.path.join(data_dir, f'{prefix}RatingTrain.txt')
        test_file = os.path.join(data_dir, f'{prefix}RatingTest.txt')
        neg_file = os.path.join(data_dir, f'{prefix}RatingNegative.txt')
        
        ftrain = open(train_file, 'w', encoding='utf-8')
        ftest = open(test_file, 'w', encoding='utf-8')
        fneg = open(neg_file, 'w', encoding='utf-8')
        
        all_items = list(item2id.values())
        
        count = 0
        
        # interactions: dict {id: [iid1, iid2, ...]}
        for obj_id, items in interactions.items():
            # Deduplicate while preserving order
            items = list(dict.fromkeys(items)) 
            
            if len(items) < 4: # Need at least 1 train + 3 test (or less if we adapt)
                continue 
            
            # Strategy: Leave-Last-K (e.g., K=3)
            # Find the last K items that are VALID (count >= 5)
            # If not enough valid items, skip or take fewer
            
            k_test = 3
            test_candidates = []
            train_items = []
            
            # Iterate backwards to find valid test items
            idx = len(items) - 1
            while idx >= 0 and len(test_candidates) < k_test:
                curr_item = items[idx]
                if curr_item in valid_test_items:
                    test_candidates.append(curr_item)
                else:
                    # If invalid for test, put it in train? Or ignore?
                    # Usually better to put in train to keep context, 
                    # but if it's rare, maybe it doesn't matter.
                    # Let's put in train if we don't use it for test.
                    train_items.append(curr_item)
                idx -= 1
            
            # Remaining items go to train
            while idx >= 0:
                train_items.append(items[idx])
                idx -= 1
                
            # Restore order for train_items (they were added backwards)
            train_items.reverse()
            # test_candidates are also backwards (newest first)
            
            if not test_candidates:
                continue
                
            # Write Train
            for ti in train_items:
                ftrain.write(f'{obj_id} {ti} 1\n')
                
            # Write Test
            # For each test item, we write a line? Or one line with multiple GT?
            # Standard evaluation (like metrics.py) often expects one line per test instance.
            # evaluate() in metrics.py loops over test_ratings:
            # rating = test_ratings[idx] -> [user, item]
            # It evaluates each (user, item) pair separately against negatives.
            # So we can write multiple lines for the same user.
            
            for test_item in test_candidates:
                ftest.write(f'{obj_id} {test_item}\n')
                
                # Write Negative
                negs = set()
                while len(negs) < 100:
                    n = random.choice(all_items)
                    if n not in items: # Not in ANY observed items (train or test)
                        negs.add(n)
                
                neg_str = ' '.join(map(str, list(negs)))
                fneg.write(f'({obj_id},{test_item}) {neg_str}\n')
            
            count += 1
            
        ftrain.close()
        ftest.close()
        fneg.close()
        print(f"Saved {count} {prefix} sequences (with Leave-Last-{k_test}).")

    save_interactions(user_interactions, 'user')
    save_interactions(group_interactions, 'group')
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    preprocess("data/facebook")
