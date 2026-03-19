#!/bin/bash

# 默认参数配置
DATASET="facebook_stopwords"
DEVICE="cuda:4" # 如果显卡不可用，可以修改为 cpu 或 cuda:0
CHECKPOINT="facebook_stopwords/model_facebook_stopwords_cl0.001_temp0.2.pth" # checkpoints/ 目录下的模型文件
RECENT_K=50
EMB_DIM=32
BGE_PATH="/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/checkpoint/bge-large-zh"

# 打印帮助信息
usage() {
    echo "使用方法: $0 [测试模式]"
    echo "支持的测试模式:"
    echo "  all     - 默认模式，运行完整测试集 (基于 data/$DATASET/groupRatingTest.txt 进行推断)"
    echo "  json    - 运行单条 JSON 样本测试 (基于 test_group_inference.json 进行推断)"
    echo "  help    - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./test.sh all"
    echo "  ./test.sh json"
}

# 获取模式参数，默认为 all
MODE=${1:-all}

if [ "$MODE" == "all" ]; then
    echo "================================================="
    echo "正在运行全量测试集推断..."
    echo "================================================="
    python inference.py \
        --dataset $DATASET \
        --device $DEVICE \
        --checkpoint $CHECKPOINT \
        --recent_k $RECENT_K \
        --emb_dim $EMB_DIM \
        --bge_path $BGE_PATH

elif [ "$MODE" == "json" ]; then
    JSON_FILE="test_group_inference.json"
    if [ ! -f "$JSON_FILE" ]; then
        echo "错误: 未找到 JSON 测试文件 $JSON_FILE"
        exit 1
    fi
        
    echo "================================================="
    echo "正在运行 JSON 样本推断 ($JSON_FILE)..."
    echo "================================================="
    python inference.py \
        --dataset $DATASET \
        --device $DEVICE \
        --checkpoint $CHECKPOINT \
        --recent_k $RECENT_K \
        --emb_dim $EMB_DIM \
        --bge_path $BGE_PATH \
        --json_input $JSON_FILE

elif [ "$MODE" == "help" ]; then
    usage
else
    echo "未知模式: $MODE"
    usage
    exit 1
fi
