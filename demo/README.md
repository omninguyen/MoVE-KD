# Demo hỏi đáp ảnh: MoVE-KD-v1.1-7B và LLaVA-1.5-7B

Thư mục này gồm hai demo suy luận, cộng với các video quay giao diện so sánh hai model.

## 1. Video

Tất cả video quay cùng ảnh, cùng câu hỏi, giải mã greedy (temperature 0). Câu trả lời vì vậy cố định, chạy lại sẽ ra đúng như trong video.

| File | Nội dung |
|---|---|
| [`videos/sbs_phan12_giaodien.mp4`](videos/sbs_phan12_giaodien.mp4) | **So sánh song song** (MoVE-KD ở trên, LLaVA-1.5 ở dưới): ảnh mẫu "ủi đồ trên taxi", 5 câu POPE mà hai model trả lời khác nhau, và một câu hỏi tự gõ |
| [`videos/sbs_phan4_dem_docchu.mp4`](videos/sbs_phan4_dem_docchu.mp4) | **So sánh song song**: 3 câu đếm (ảnh COCO trong mục count của MME) và 4 câu đọc chữ (ảnh TextVQA, hỏi **không** kèm OCR token) |
| `videos/phan1_movekd_giaodien.mp4`, `videos/phan2_llava15_giaodien.mp4` | Bản gốc của từng model, dùng để ghép video so sánh thứ nhất |
| `videos/phan4_movekd_dem_docchu.mp4`, `videos/phan4_llava15_dem_docchu.mp4` | Bản gốc của từng model, dùng để ghép video so sánh thứ hai |

Trong video so sánh, đoạn chờ model sinh câu trả lời được tua nhanh 2 lần (có nhãn "» 2x" bên trái). Câu hỏi thứ i của hai model bắt đầu cùng lúc.

**Lưu ý khi đọc video:**
- Các câu POPE được **chọn** từ những câu mà hai model trả lời khác nhau khi chạy full benchmark. Trong đó có 2 câu MoVE-KD trả lời sai (toothbrush, bench). Video chỉ để **minh hoạ**. Bằng chứng định lượng là bảng kết quả trong [`../Experiments.md`](../Experiments.md) mục 2.5.
- Video đọc chữ hỏi không kèm OCR token. Benchmark TextVQA trong eval của LLaVA thì **có** kèm danh sách OCR token trong prompt. Hai điều kiện này khác nhau.

## 2. Giao diện so sánh dùng để quay video: `scripts/move-kd/demo_ui.py`

Đây là giao diện Gradio nạp một checkpoint. Ảnh và câu hỏi mẫu được đặt sẵn: 2 ảnh mẫu của LLaVA, 5 câu POPE kèm nhãn đúng, 7 câu đếm và đọc chữ.

```bash
# MoVE-KD (từ thư mục gốc repo này)
python scripts/move-kd/demo_ui.py --model-path checkpoints/llava-move-kd-7b-v1.1

# LLaVA-1.5 gốc: chạy bằng code LLaVA gốc như scripts/move-kd/eval/baseline.sh
PYTHONPATH=/path/to/LLaVA python scripts/move-kd/demo_ui.py --model-path checkpoints/llava-v1.5-7b
```

- Prompt của các câu POPE giữ nguyên như khi chạy eval, nên câu trả lời khớp với lần chạy full benchmark.
- Ảnh được đưa thẳng vào model, không qua bước lưu lại của Gradio. Gradio 4.16 nén JPEG thêm một lần khi lưu ảnh, và chừng đó đủ làm lật câu trả lời của LLaVA-1.5 ở các câu POPE sát ngưỡng.
- Ảnh đếm và đọc chữ không có trong repo. Cách tải được ghi ở đầu `demo_ui.py`: COCO val2017 và TextVQA val, lưu vào `playground/data/demo_extra/`. Nếu thiếu các ảnh này, giao diện tự ẩn nhóm ví dụ đó.

Hai script đi kèm:
- `scripts/move-kd/demo_ask.py`: nạp model một lần rồi trả lời một danh sách câu hỏi (`demo_questions.json`), hoặc hỏi tương tác.
- `scripts/move-kd/demo_compare.py`: liệt kê các câu POPE mà hai model trả lời khác nhau (dùng để chọn ví dụ), và đặt cạnh nhau câu trả lời của hai model.

## 3. Demo suy luận đơn giản: `demo/move_kd_demo.py`

Demo do thành viên khác trong nhóm viết: giao diện Gradio có stream câu trả lời, hoặc hỏi nhanh từ dòng lệnh. Xem phần hướng dẫn ở đầu file.

```bash
python demo/move_kd_demo.py                                   # giao diện web, localhost:7860
python demo/move_kd_demo.py --image a.jpg --question "What is in the image?"
```

## Yêu cầu

- Linux, GPU NVIDIA. Model 7B ở fp16 dùng khoảng 15GB VRAM.
- Môi trường cài theo `setup.sh`.
- Checkpoint tải bằng `bash scripts/move-kd/eval/download_ckpt.sh`. Tên thư mục checkpoint phải chứa chữ `llava`, xem `Experiments.md` mục 2.6.
