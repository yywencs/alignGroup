import opencc
import os

def append_traditional_stopwords(file_path):
    # Initialize converter
    # s2t.json is for Simplified Chinese to Traditional Chinese
    try:
        cc = opencc.OpenCC('s2t')
    except Exception as e:
        print(f"Error initializing OpenCC: {e}")
        return

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Read existing stopwords
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Use a set for faster lookup
    existing_words = set(line.strip() for line in lines)
    new_words = set()
    
    print(f"Read {len(existing_words)} unique existing stopwords.")

    # Convert and find new traditional words
    for word in existing_words:
        if not word:
            continue
            
        traditional_word = cc.convert(word)
        
        # If the converted word is different and not already in the list
        if traditional_word != word and traditional_word not in existing_words:
            new_words.add(traditional_word)
    
    # Convert set back to list and sort
    new_words_list = sorted(list(new_words))
    
    if new_words_list:
        print(f"Found {len(new_words_list)} new traditional Chinese stopwords.")
        with open(file_path, 'a', encoding='utf-8') as f:
            # Ensure we start on a new line if the file doesn't end with one
            if lines and lines[-1].strip() and not lines[-1].endswith('\n'):
                f.write('\n')
            
            for word in new_words_list:
                f.write(word + '\n')
        print(f"Successfully appended {len(new_words_list)} words to {file_path}.")
    else:
        print("No new traditional Chinese stopwords found.")

if __name__ == "__main__":
    file_path = '/Users/wen/Documents/项目/code/AlignGroup/stopwords_baidu.txt'
    append_traditional_stopwords(file_path)
