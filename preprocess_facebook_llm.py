import os
import csv
import re
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Set random seed
random.seed(2025)

def get_llm_keywords_batch(texts, model, tokenizer, device):
    """
    使用 LLM 批量提取关键词
    """
    prompts = []
    for t in texts:
        # 清洗一下基础噪音
        t_clean = re.sub(r'http\S+', '', t)
        t_clean = re.sub(r'\s+', ' ', t_clean).strip()
        
        if not t_clean:
            prompts.append(None) # 标记空文本
            continue
            
        # Few-shot Prompt 设计
        prompt = (
            "任务：从以下文本中提取有实质意义的关键词（如实体、名词、动词），去除无意义的语气词（如哈哈、嗯嗯、那个）和常用停用词、时态等。\n"
            "要求：直接输出关键词，用空格分隔，不要输出任何解释或其他内容。\n\n"
            "例子：\n"
            "文本：今天天气真好呀，哈哈，我们要去公园野餐，记得带上风筝。\n"
            "关键词：公园 野餐 风筝\n\n"
            f"文本：{t_clean}\n"
            "关键词："
        )
        prompts.append(prompt)
    
    # 过滤掉空的 prompt
    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [prompts[i] for i in valid_indices]
    
    if not valid_prompts:
        return [""] * len(texts)
    
    # Tokenize
    tokenizer.padding_side = 'left' # Decoder-only 模型 batch 推理需要左填充
    inputs = tokenizer(valid_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=64, # 关键词一般不会太长
            do_sample=False,   # 使用贪婪搜索，保证结果确定且速度快
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    # Decode (只解码新生成的部分)
    input_len = inputs.input_ids.shape[1]
    new_tokens = generated_ids[:, input_len:]
    outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    # 还原结果顺序
    final_results = [""] * len(texts)
    for idx, out in zip(valid_indices, outputs):
        final_results[idx] = out.strip()
        
    return final_results

def preprocess(data_dir, model_path, out_dir=None, limit=0, batch_size=16):
    print("Processing Facebook dataset with LLM...")
    csv_path = os.path.join(data_dir, "data_merged.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    if out_dir is None:
        out_dir = data_dir

    # Load LLM
    print(f"Loading Qwen model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Mappings
    user2id = {}
    group2id = {}
    item2id = {}

    user_interactions = {}
    group_interactions = {}
    group_members = {}

    raw_data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            raw_data.append(row)

    def parse_date(d_str):
        try:
            return d_str
        except:
            return "0000/00/00/00/00"

    raw_data.sort(key=lambda x: parse_date(x.get('Date Posted', '')))

    if limit and limit > 0:
        print(f"Limiting to first {limit} rows for testing.")
        raw_data = raw_data[:limit]

    print(f"Total rows: {len(raw_data)}")

    total_rows = len(raw_data)
    pbar = tqdm(total=total_rows, desc="LLM 关键词提取", unit="句", dynamic_ncols=True)
    start_time = time.time()

    for i in range(0, total_rows, batch_size):
        batch_rows = raw_data[i:i + batch_size]
        texts = []
        for row in batch_rows:
            content = (row.get('Post Content', '') or '').strip()
            content = re.sub(r'http\S+', '', content)
            texts.append(content)

        keywords_batch = get_llm_keywords_batch(texts, model, tokenizer, model.device)

        for row, kw_str in zip(batch_rows, keywords_batch):
            g_name = (row.get('Group Name', '') or '').strip()
            u_name = (row.get('User Name', '') or '').strip()

            if not g_name or not u_name or not kw_str:
                continue

            if g_name not in group2id:
                group2id[g_name] = len(group2id)
            gid = group2id[g_name]

            if u_name not in user2id:
                user2id[u_name] = len(user2id)
            uid = user2id[u_name]

            if gid not in group_members:
                group_members[gid] = set()
            group_members[gid].add(uid)

            words = kw_str.split()
            for w in words:
                w = w.strip()
                if len(w) < 2:
                    continue

                if w not in item2id:
                    item2id[w] = len(item2id)
                iid = item2id[w]

                if uid not in user_interactions:
                    user_interactions[uid] = []
                user_interactions[uid].append(iid)

                if gid not in group_interactions:
                    group_interactions[gid] = []
                group_interactions[gid].append(iid)

        pbar.update(len(batch_rows))
        elapsed = time.time() - start_time
        if elapsed > 0:
            pbar.set_postfix_str(f"{pbar.n/elapsed:.2f} 句/s")

    pbar.close()

    print(f"Users: {len(user2id)}")
    print(f"Groups: {len(group2id)}")

    item_counts = {}
    for _, items in user_interactions.items():
        for iid in items:
            item_counts[iid] = item_counts.get(iid, 0) + 1
    for _, items in group_interactions.items():
        for iid in items:
            item_counts[iid] = item_counts.get(iid, 0) + 1

    print(f"Items (LLM Keywords): {len(item2id)}")

    valid_test_items = set([iid for iid, c in item_counts.items() if c >= 5])
    print(f"Valid Test Items (Count >= 5): {len(valid_test_items)}")

    with open(os.path.join(out_dir, 'group_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in group2id.items():
            f.write(f'{v} {k}\n')

    with open(os.path.join(out_dir, 'user_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in user2id.items():
            f.write(f'{v} {k}\n')

    with open(os.path.join(out_dir, 'item_list.txt'), 'w', encoding='utf-8') as f:
        for k, v in item2id.items():
            f.write(f'{v} {k}\n')

    with open(os.path.join(out_dir, 'groupMember.txt'), 'w', encoding='utf-8') as f:
        for gid in range(len(group2id)):
            if gid in group_members:
                members = group_members[gid]
                m_str = ','.join(map(str, members))
                f.write(f'{gid} {m_str}\n')
            else:
                f.write(f'{gid} \n')

    def save_interactions(interactions, prefix):
        print(f"Saving {prefix} interactions...")

        train_file = os.path.join(out_dir, f'{prefix}RatingTrain.txt')
        test_file = os.path.join(out_dir, f'{prefix}RatingTest.txt')
        neg_file = os.path.join(out_dir, f'{prefix}RatingNegative.txt')

        ftrain = open(train_file, 'w', encoding='utf-8')
        ftest = open(test_file, 'w', encoding='utf-8')
        fneg = open(neg_file, 'w', encoding='utf-8')

        all_items = list(item2id.values())
        count = 0
        k_test = 3

        for obj_id, items in interactions.items():
            items = list(dict.fromkeys(items))
            if len(items) < 4:
                continue

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

            train_items.reverse()

            if not test_candidates:
                continue

            for ti in train_items:
                ftrain.write(f'{obj_id} {ti} 1\n')

            for test_item in test_candidates:
                ftest.write(f'{obj_id} {test_item}\n')

                negs = set()
                while len(negs) < 100:
                    n = random.choice(all_items)
                    if n not in items:
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

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Facebook dataset with LLM")
    parser.add_argument("--data_dir", type=str, default="/home/yangyw/code/my_code/rz/AlignGroup/data/facebook", help="Data directory")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: same as data_dir)")
    parser.add_argument("--model_path", type=str, default="/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/checkpoint/Qwen3-1.7B", help="Model path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows for testing")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for LLM inference")
    
    args = parser.parse_args()
    
    out_dir = args.out_dir.strip() or None
    preprocess(args.data_dir, args.model_path, out_dir=out_dir, limit=args.limit, batch_size=args.batch_size)
                
