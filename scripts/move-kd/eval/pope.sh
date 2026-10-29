#!/bin/bash
# POPE (ảo giác đối tượng, COCO val2014). Dữ liệu: playground/data/eval/pope/{llava_pope_test.jsonl,coco/,val2014/}
# Dùng: [LIMIT=50] bash scripts/move-kd/eval/pope.sh [CKPT]
source "$(dirname "$0")/common.sh"
DIR=./playground/data/eval/pope
QUESTIONS="$(subset_jsonl $DIR/llava_pope_test.jsonl)"

python -m llava.eval.model_vqa_loader \
    --model-path "$CKPT" \
    --question-file "$QUESTIONS" \
    --image-folder $DIR/val2014 \
    --answers-file $DIR/answers/$TAG.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python llava/eval/eval_pope.py \
    --annotation-dir $DIR/coco \
    --question-file "$QUESTIONS" \
    --result-file $DIR/answers/$TAG.jsonl | tee $DIR/answers/${TAG}_score.txt
