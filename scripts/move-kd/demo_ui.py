"""Giao diện web hỏi đáp ảnh cho video demo (Gradio), nạp một checkpoint duy nhất.

    python scripts/move-kd/demo_ui.py --model-path checkpoints/llava-move-kd-7b-v1.1 [--port 7860]
    # LLaVA-1.5 gốc: chạy từ code LLaVA như baseline.sh
    PYTHONPATH=/workspace/LLaVA python scripts/move-kd/demo_ui.py --model-path checkpoints/llava-v1.5-7b

Ví dụ POPE lấy từ các câu mà MoVE-KD và LLaVA-1.5 trả lời khác nhau (demo_compare.py pope), giữ nguyên prompt
của eval để kết quả khớp với lần chạy full.
"""
import argparse
import os
import sys
import time

import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_ask import answer  # noqa: E402
from llava.mm_utils import get_model_name_from_path  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

POPE = "playground/data/eval/pope/val2014"
SUFFIX = "\nAnswer the question using a single word or phrase."
# (ảnh, câu hỏi, ghi chú): nhãn POPE và câu trả lời đã chạy full của hai model
EXAMPLES = [
    ("llava/serve/examples/extreme_ironing.jpg", "What is unusual about this image?", "Ảnh mẫu của LLaVA"),
    ("llava/serve/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?", "Ảnh mẫu của LLaVA"),
    (f"{POPE}/COCO_val2014_000000310196.jpg", "Is there a skis in the image?" + SUFFIX, "POPE, nhãn yes · MoVE-KD: Yes · LLaVA-1.5: No"),
    (f"{POPE}/COCO_val2014_000000429109.jpg", "Is there a truck in the image?" + SUFFIX, "POPE, nhãn no · MoVE-KD: No · LLaVA-1.5: Yes"),
    (f"{POPE}/COCO_val2014_000000283412.jpg", "Is there a book in the image?" + SUFFIX, "POPE, nhãn no · MoVE-KD: No · LLaVA-1.5: Yes"),
    (f"{POPE}/COCO_val2014_000000542145.jpg", "Is there a toothbrush in the image?" + SUFFIX, "POPE, nhãn yes · MoVE-KD: No · LLaVA-1.5: Yes (MoVE-KD sai)"),
    (f"{POPE}/COCO_val2014_000000131115.jpg", "Is there a bench in the image?" + SUFFIX, "POPE, nhãn no · MoVE-KD: Yes · LLaVA-1.5: No (MoVE-KD sai)"),
]
# Đếm (ảnh COCO val2017 của mục count trong MME) và đọc chữ (ảnh TextVQA val, hỏi KHÔNG kèm OCR token).
# Tải về playground/data/demo_extra/: images.cocodataset.org/val2017/<id>.jpg và train_images/<id>.jpg trong
# dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip (đặt tên tvqa_<id>.jpg).
EXTRA = "playground/data/demo_extra"
EXAMPLES_EXTRA = [
    (f"{EXTRA}/000000097994.jpg", "How many laptops are in the picture?", "Đếm · đáp án: 3 (MME count)"),
    (f"{EXTRA}/000000430286.jpg", "How many remotes are in the image?", "Đếm · đáp án: 3 (MME count)"),
    (f"{EXTRA}/000000357816.jpg", "How many people are in this image?", "Đếm · đáp án: 4 (MME count)"),
    (f"{EXTRA}/tvqa_2b92ce942467f122.jpg", "What is the brand name of the green bottled beer?", "Đọc chữ · đáp án: Tsingtao (TextVQA)"),
    (f"{EXTRA}/tvqa_0442ab5ce00a56fa.jpg", "What brand is this time piece?", "Đọc chữ · đáp án: Rolex (TextVQA)"),
    (f"{EXTRA}/tvqa_d222362444d7f282.jpg", "What brand of drink is in the bottle?", "Đọc chữ · đáp án: Perrier (TextVQA)"),
    (f"{EXTRA}/tvqa_0bfd8d9d21774a40.jpg", "What's the word after 'Olivetti'?", "Đọc chữ · đáp án: Underwood (TextVQA)"),
]


class RawImage(gr.Image):
    """gr.Image trả về đúng file người dùng đưa vào. Image.preprocess của Gradio 4.16 mở ảnh rồi lưu lại
    (JPEG nén thêm một lần), đủ làm lật câu trả lời của LLaVA-1.5 ở các câu POPE sát ngưỡng."""

    is_template = True  # frontend dùng component "image" của lớp cha

    def preprocess(self, payload):
        return None if payload is None else str(payload.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    disable_torch_init()
    name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, name)
    info = f"**Model:** `{name}` · lớp `{type(model).__name__}` · VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB · giải mã greedy"
    print(info, flush=True)

    count = [0]

    def run(image, question):
        if image is None or not question.strip():
            return "Chọn ảnh và nhập câu hỏi.", ""
        t = time.time()
        a = answer(tokenizer, model, image_processor, image, question.strip(), args.max_new_tokens)
        count[0] += 1
        return a, f"{time.time() - t:.1f} giây · câu #{count[0]}"

    with gr.Blocks(title=f"MoVE-KD demo · {name}") as ui:
        gr.Markdown(f"## Hỏi đáp ảnh với {name}\n{info}")
        with gr.Row():
            with gr.Column(scale=1):
                img = RawImage(type="filepath", label="Ảnh", height=330)
                q = gr.Textbox(label="Câu hỏi", lines=2)
                note = gr.Textbox(label="Ghi chú ví dụ", interactive=False)
                btn = gr.Button("Hỏi model", variant="primary")
            with gr.Column(scale=1):
                out = gr.Textbox(label="Câu trả lời", lines=9)
                dt = gr.Textbox(label="Thời gian", interactive=False)
        gr.Examples([list(e) for e in EXAMPLES if os.path.exists(e[0])], inputs=[img, q, note], label="Ví dụ")
        extra = [list(e) for e in EXAMPLES_EXTRA if os.path.exists(e[0])]
        if extra:
            gr.Examples(extra, inputs=[img, q, note], label="Đếm & đọc chữ")
        btn.click(run, [img, q], [out, dt], api_name="ask")
        q.submit(run, [img, q], [out, dt])
    ui.queue().launch(server_name="127.0.0.1", server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
