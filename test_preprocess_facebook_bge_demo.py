import argparse

from preprocess_facebook_bge import demo_bge_output, extract_keywords_from_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/checkpoint/bge-large-zh",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="15分鐘搞懂 Agent Skills 是什麼，怎麼用",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    demo_bge_output(args.model_path, args.text)
    result = extract_keywords_from_text(
        args.model_path,
        args.text,
        similarity_threshold=args.threshold,
    )
    print("content_clean:", result["content_clean"])
    print("jieba_words:", result["jieba_words"])
    print("candidates:", result["candidates"])
    print("unique_candidates:", result["unique_candidates"])
    print("selected:", result["selected"])
    print("top_scored:", result["scored"][: max(0, args.topk)])


if __name__ == "__main__":
    main()
