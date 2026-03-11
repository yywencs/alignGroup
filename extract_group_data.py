import csv
import json
import random
import os

csv_file_path = '/home/yangyw/code/my_code/rz/AlignGroup/data/data_merged.csv'
output_json_path = '/home/yangyw/code/my_code/rz/AlignGroup/test_group_inference.json'

def extract_group_data():
    # 1. Count posts per group
    group_posts = {} # group_name -> list of post contents
    
    print("Reading CSV file...")
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Header: Group Name,User Name,Post Content,...
        
        for row in reader:
            if len(row) < 3:
                continue
                
            group_name = row[0]
            post_content = row[2]
            
            if not group_name or not post_content:
                continue
                
            if group_name not in group_posts:
                group_posts[group_name] = []
            
            group_posts[group_name].append(post_content)
            
    print(f"Found {len(group_posts)} groups.")
    
    # 2. Filter groups with enough posts (e.g., >= 50)
    candidates = {g: posts for g, posts in group_posts.items() if len(posts) >= 50}
    
    if not candidates:
        print("No group has >= 50 posts. Relaxing criteria to >= 20.")
        candidates = {g: posts for g, posts in group_posts.items() if len(posts) >= 20}
        
    if not candidates:
        print("No suitable groups found.")
        return

    # 3. Select a group
    # Prefer a group with a reasonable name length (not too weird)
    selected_group = None
    selected_posts = []
    
    # Sort by post count descending
    sorted_groups = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Pick the top one
    selected_group, all_posts = sorted_groups[0]
    
    print(f"Selected Group: {selected_group} with {len(all_posts)} posts.")
    
    # 4. Extract 50-100 posts
    # If more than 100, take recent 100 (assuming CSV order is somewhat chronological or random, just taking last 100)
    # Actually, let's take the first 100 to simulate 'history'
    num_posts = min(len(all_posts), 100)
    chat_data = all_posts[:num_posts]
    
    # 5. Generate JSON
    # Description might not be in CSV, so we leave it empty or generic
    data = {
        "group_name": selected_group,
        "group_description": f"Community for {selected_group}",
        "chat_data": chat_data
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"JSON generated at {output_json_path}")
    print(f"Sample chat data: {chat_data[:2]}")

if __name__ == "__main__":
    extract_group_data()
