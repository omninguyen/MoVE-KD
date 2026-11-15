"""Demo hỏi đáp về ảnh với MoVE-KD (chỉ suy luận, không huấn luyện).

Hai chế độ:
  # Giao diện web (Gradio), mặc định chỉ lắng nghe ở localhost:7860
  python demo/move_kd_demo.py

  # Hỏi nhanh từ dòng lệnh, in câu trả lời rồi thoát
  python demo/move_kd_demo.py --image a.jpg --question "What is in the image?"

Yêu cầu: chạy `pip install -e .` ở thư mục gốc repo và có GPU NVIDIA (CUDA).
"""
import argparse
from threading import Thread

import gradio as gr
import torch
from PIL import Image
from transformers import TextIteratorStreamer

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# Checkpoint được huấn luyện với `--version v1` (xem scripts/move-kd/finetune-v1.1.sh),
# tức template vicuna_v1. Không suy ra template từ tên model: "MoVE-KD-7b-v1.1" chứa "v1"
# nên sẽ bị gán nhầm sang llava_v1.
CONV_MODE = "vicuna_v1"


def load_model(model_path, load_8bit=False, load_4bit=False):
    disable_torch_init()
    model_name = model_path.strip("/").split("/")[-1]
    # load_pretrained_model chỉ tạo vision tower và image processor khi tên model chứa "llava".
    # Thêm tiền tố để không nhận image_processor = None.
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, f"llava-{model_name}",
        load_8bit=load_8bit, load_4bit=load_4bit, device="cuda",
    )
    model.eval()
    return tokenizer, model, image_processor


def build_input_ids(question, tokenizer, model):
    conv = conv_templates[CONV_MODE].copy()
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return input_ids.unsqueeze(0).to(model.device)


def preprocess_image(image, image_processor, model):
    # Cùng cách tiền xử lý với llava/serve/cli.py: pad thành hình vuông rồi resize về 336x336
    image_tensor = process_images([image.convert("RGB")], image_processor, model.config)
    return image_tensor.to(model.device, dtype=torch.float16)


def stream_answer(model, tokenizer, image_tensor, image_size, input_ids, temperature, max_new_tokens):
    """Sinh câu trả lời trong thread riêng và yield văn bản tích luỹ để UI hiện dần."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        images=image_tensor,
        image_sizes=[image_size],
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)

    errors = []

    def run():
        try:
            with torch.inference_mode():
                model.generate(input_ids, **gen_kwargs)
        except Exception as e:
            errors.append(e)
        finally:
            # Đảm bảo vòng lặp bên dưới thoát ngay cả khi generate lỗi giữa chừng
            streamer.end()

    thread = Thread(target=run)
    thread.start()
    text = ""
    for piece in streamer:
        text += piece
        yield text.strip()
    thread.join()
    if errors:
        raise errors[0]


def build_ui(tokenizer, model, image_processor):
    def answer(image, question, temperature, max_new_tokens):
        if image is None:
            raise gr.Error("Vui lòng tải lên một ảnh.")
        if not question or not question.strip():
            raise gr.Error("Vui lòng nhập câu hỏi.")
        image_tensor = preprocess_image(image, image_processor, model)
        input_ids = build_input_ids(question.strip(), tokenizer, model)
        yield from stream_answer(
            model, tokenizer, image_tensor, image.size, input_ids,
            temperature, int(max_new_tokens),
        )

    with gr.Blocks(title="MoVE-KD Demo") as demo:
        gr.Markdown(
            "# MoVE-KD: hỏi đáp về ảnh\n"
            "Tải ảnh lên rồi đặt câu hỏi. Nên hỏi bằng tiếng Anh vì mô hình được huấn luyện chủ yếu trên dữ liệu tiếng Anh."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Ảnh")
                question = gr.Textbox(
                    label="Câu hỏi", lines=2,
                    placeholder="What is happening in this image?",
                )
                with gr.Accordion("Tham số sinh", open=False):
                    temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1,
                                            label="Temperature (0 = greedy, ổn định nhất)")
                    max_new_tokens = gr.Slider(16, 1024, value=512, step=16,
                                               label="Số token tối đa")
                submit = gr.Button("Hỏi", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Trả lời", lines=14, interactive=False)

        submit.click(answer, inputs=[image, question, temperature, max_new_tokens], outputs=output)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="jiaojuncao/MoVE-KD-7b-v1.1",
                        help="Repo HF hoặc thư mục checkpoint đã tải về")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image", type=str, default=None, help="Chỉ dùng cho chế độ CLI")
    parser.add_argument("--question", type=str, default=None, help="Có đặt thì chạy chế độ CLI")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Tạo link công khai của Gradio (ảnh sẽ đi qua máy chủ Gradio)")
    args = parser.parse_args()

    if args.question and not args.image:
        parser.error("--question cần đi kèm --image")

    tokenizer, model, image_processor = load_model(args.model_path, args.load_8bit, args.load_4bit)

    if args.question:
        image = Image.open(args.image).convert("RGB")
        image_tensor = preprocess_image(image, image_processor, model)
        input_ids = build_input_ids(args.question, tokenizer, model)
        printed = ""
        for text in stream_answer(model, tokenizer, image_tensor, image.size, input_ids,
                                  args.temperature, args.max_new_tokens):
            print(text[len(printed):], end="", flush=True)
            printed = text
        print()
        return

    demo = build_ui(tokenizer, model, image_processor)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
