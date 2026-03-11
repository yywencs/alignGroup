import os
from collections import Counter

data_dir = "/home/yangyw/code/my_code/rz/AlignGroup/data/facebook"
item_list_path = os.path.join(data_dir, "item_list.txt")
group_test_path = os.path.join(data_dir, "groupRatingTest.txt")

# 1. Load Item List
print("Loading item list...")
id2word = {}
with open(item_list_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            iid = int(parts[0])
            word = parts[1]
            id2word[iid] = word

print(f"Loaded {len(id2word)} items.")

# 2. Load Group Test Data
print("Loading group test data...")
test_words = []
with open(group_test_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            iid = int(parts[1])
            if iid in id2word:
                test_words.append(id2word[iid])

# 3. Analyze
print(f"Total test interactions: {len(test_words)}")
counter = Counter(test_words)

print("\nTop 50 most frequent words in Group Test Set:")
print("-" * 40)
print(f"{'Word':<15} | {'Count':<10} | {'Percentage'}")
print("-" * 40)

total = len(test_words)
for word, count in counter.most_common(50):
    percent = (count / total) * 100
    print(f"{word:<15} | {count:<10} | {percent:.2f}%")

print("-" * 40)