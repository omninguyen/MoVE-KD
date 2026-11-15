"""Hỏi đáp ảnh với một checkpoint, nạp model một lần rồi trả lời nhiều câu (dùng cho video demo).

    python scripts/move-kd/demo_ask.py --model-path checkpoints/llava-move-kd-7b-v1.1 \
        --questions scripts/move-kd/demo_questions.json --out demo_out/move-kd.json
    python scripts/move-kd/demo_ask.py --model-path ... --interactive     # gõ: <đường dẫn ảnh> | <câu hỏi>

Chạy được cả với LLaVA gốc (từ checkout LLaVA, như baseline.sh), vì chỉ dùng API chung của package llava.
Giải mã greedy (temperature 0) giống các script eval, nên chạy lại cho cùng câu trả lời.
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def answer(tokenizer, model, image_processor, image_path, question, max_new_tokens):
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    # nhận đường dẫn hoặc ảnh PIL
    image = (image_path if isinstance(image_path, Image.Image) else Image.open(image_path)).convert("RGB")
    images = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    with torch.inference_mode():
        out = model.generate(input_ids, images=images, image_sizes=[image.size], do_sample=False,
                             max_new_tokens=max_new_tokens, use_cache=True)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--questions", help="file JSON: danh sách {image, question}")
    ap.add_argument("--out", help="ghi kết quả ra file JSON để so sánh hai model")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    disable_torch_init()
    name = get_model_name_from_path(args.model_path)
    t0 = time.time()
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, name)
    print(f"\n=== {name}: nạp model {time.time() - t0:.0f}s, {type(model).__name__}, "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB ===\n", flush=True)

    results = []
    if args.questions:
        for i, q in enumerate(json.load(open(args.questions)), 1):
            t = time.time()
            a = answer(tokenizer, model, image_processor, q["image"], q["question"], args.max_new_tokens)
            print(f"[{i}] {os.path.basename(q['image'])}\n  Hỏi: {q['question']}\n  Đáp: {a}\n  ({time.time() - t:.1f}s)\n", flush=True)
            results.append({**q, "answer": a})
    if args.interactive:
        print("Nhập '<đường dẫn ảnh> | <câu hỏi>', dòng trống để thoát.")
        while True:
            line = input("> ").strip()
            if not line:
                break
            img, q = (s.strip() for s in line.split("|", 1))
            print("  Đáp:", answer(tokenizer, model, image_processor, img, q, args.max_new_tokens), "\n", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"model": name, "results": results}, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
