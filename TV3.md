# Phần việc của Thành viên 3: Thực nghiệm, Benchmark và chạy code

Phân công: Sec 4 của paper (dữ liệu, benchmark, ablation), tự chạy code (eval, trực quan hoá attention giống Fig 5) và viết Related Work cùng Experiments & Results.

Tài liệu chính: [`Experiments.md`](Experiments.md), bản LaTeX tương ứng là [`report/tv3_report.pdf`](report/tv3_report.pdf).

---

## 1. Tóm tắt kết quả

Mọi phép đo dùng **checkpoint tác giả công bố** ([`jiaojuncao/MoVE-KD-7b-v1.1`](https://huggingface.co/jiaojuncao/MoVE-KD-7b-v1.1)), chạy trên **1× RTX 3090 24GB**, fp16, giải mã greedy. Paper huấn luyện trên cụm 16 GPU, nên nhóm không huấn luyện lại.

| Benchmark | LLaVA-1.5-7B (paper) | LLaVA-1.5-7B (nhóm chạy) | MoVE-KD-v1.1-7B (paper) | MoVE-KD-v1.1-7B (nhóm chạy) |
|---|---:|---:|---:|---:|
| POPE (F1 TB 3 split) | 85.9 | 85.92 | 86.3 | 86.06 |
| MME (perception) | 1510.7 | 1504.6 | 1509.1 | 1510.0 |
| TextVQA | 58.2 | 58.23 | 59.6 | 59.62 |
| SQA-IMG | **66.8** | **69.46** | 69.8 | 69.81 |

Bốn phát hiện chính:

1. **Tái lập được checkpoint.** MoVE-KD tự chạy lệch paper dưới 0.3 điểm ở cả 4 benchmark (MME lệch 0.9 trên thang khoảng 2000).
2. **Mức tăng SQA nhỏ hơn nhiều khi so cùng điều kiện.** Baseline LLaVA-1.5 nhóm chạy được 69.46, trong khi paper ghi 66.8. Vì vậy mức tăng SQA giảm từ +3.0 (paper) xuống **+0.35**, và mức +0.35 này không có ý nghĩa thống kê: hơn 7 câu trên 2017, McNemar p ≈ 0.64. Mức tăng rõ nhất còn lại là **TextVQA +1.4**, khớp với paper.
3. **Bên trong mô hình:**
   - Student giữ gần nguyên bản đồ [CLS] attention của CLIP. Các đỉnh attention nằm ở nền và dải padding, không nằm trên vật thể.
   - Router của MoLE **không chia token theo nội dung**: layer nông trộn hai expert, nhiều layer sâu dồn 70–100% token vào một expert. Nguyên nhân đọc được trong code (ở chế độ sparse router không nhận gradient từ loss chính; balance loss gộp cả 24 layer). Hình routing này paper không có.
4. **Code huấn luyện công bố bị crash ngay bước đầu.** Nhóm tìm ra nguyên nhân và sửa bằng một dòng (commit `84fe10e`), sau đó chạy được Stage 1 với đủ 4 teacher.

Chi tiết và phân tích: `Experiments.md` mục 2.4–2.6.

---

## 2. Danh sách việc đã làm và file tương ứng

### 2.1 Eval trên 4 benchmark chấm được tại máy
| File | Nội dung |
|---|---|
| `scripts/move-kd/eval/download_ckpt.sh` | Tải checkpoint vào thư mục có tên chứa `llava`. Không đặt tên như vậy thì `builder.py` nạp nhầm thành LLM thuần, không có xử lý ảnh |
| `scripts/move-kd/eval/{pope,mme,textvqa,sqa}.sh`, `common.sh` | Wrapper eval nhận đường dẫn checkpoint, hỗ trợ `LIMIT=N` để chạy thử trên tập con |
| `scripts/move-kd/eval/baseline.sh` | Chạy LLaVA-1.5 gốc bằng **code LLaVA gốc**. Code của repo này gắn expert LoRA ngẫu nhiên vào mọi checkpoint tên "llava", nên không chạy baseline được |
| `scripts/move-kd/summarize_results.py` | Đọc kết quả, sinh bảng và biểu đồ `images/results_*.png` |
| `playground/data/eval/*/answers/*_score.txt` | Điểm thô do các script chấm của LLaVA in ra, cho cả hai model |
| `notebooks/move_kd_colab.ipynb` | Notebook chạy toàn bộ trên một GPU: cài đặt, tải dữ liệu, eval |

Kiểm tra chéo bộ chấm: chấm lại câu trả lời có sẵn của LLaVA-1.5-13B (trong `eval.zip` của LLaVA) ra POPE 85.95, MME 1531.3, TextVQA 61.25, SQA 71.64, khớp Table 1 của paper.

### 2.2 Trực quan hoá
| File | Nội dung |
|---|---|
| `scripts/move-kd/visualize_attention.py` | Chỉ nạp phần vision tower (chạy được trên CPU). Vẽ [CLS] attention của CLIP gốc và student theo đúng công thức token weight trong code (vẽ lại Fig 5), và vẽ expert MoLE được chọn cho từng token ở layer 0, 8, 16, 22 |
| `images/attn_*.png`, `images/routing_*.png`, `images/routing_share.png` | Kết quả trên 6 ảnh (2 ảnh mẫu LLaVA, 2 ảnh COCO, 2 ảnh TextVQA). `routing_share.png` thống kê tỉ lệ expert ở cả 24 layer |
| `scripts/move-kd/plot_tokens.py`, `images/tokens_cls_vs_patch.png` | Hình minh hoạ patch token và token [CLS] dùng cho slide |

### 2.3 Chạy pipeline huấn luyện và sửa lỗi
| File | Nội dung |
|---|---|
| `scripts/move-kd/smoke_pretrain.sh` | Chạy Stage 1 với đủ 4 teacher, thay Vicuna-7B bằng TinyLlama-1.1B, 100 mẫu, 30 bước. Kết quả: 2 phút, VRAM đỉnh 17.8GB, loss giảm từ 10.13 xuống 4.68, lưu đủ 3 file Stage 2 cần |
| `llava/model/language_model/move_llava_llama.py` (`initialize_teachers`) | Bản sửa lỗi: `self.get_model().get_vision_tower().args.kd = self.config.kd` |

Về lỗi: lúc huấn luyện, vision tower được dựng từ tham số dòng lệnh, vốn không có trường `kd`, nên tower chỉ trả về 1 tensor. Trong khi đó `encode_images` luôn tách 2 giá trị. Với batch 2 ảnh, Python lặng lẽ tách ảnh 1 và ảnh 2 thành hai biến, làm mất chiều batch, rồi `torch.cat` báo lỗi "got 2 and 1". Lỗi ảnh hưởng cả Stage 1 và Stage 2. Eval không bị ảnh hưởng, nên các số liệu ở mục 1 không thay đổi. Chi tiết: `Experiments.md` mục 2.6 ý 10.

### 2.4 Demo
Xem [`demo/README.md`](demo/README.md): giao diện so sánh `scripts/move-kd/demo_ui.py`, các script `demo_ask.py` và `demo_compare.py`, cùng 2 video so sánh song song MoVE-KD và LLaVA-1.5 trong `demo/videos/`.

### 2.5 Báo cáo
| File | Nội dung |
|---|---|
| `Experiments.md` | Related Work, thiết lập, Table 1, ablation Table 2–5, trực quan hoá, kết quả tự chạy, ghi chú tái lập, cách tái lập |
| `report/tv3_report.tex`, `report/tv3_report.pdf` | Bản LaTeX của phần trên |
| `report/related_work.tex`, `report/related_work.bib` | Phần Related Work và tài liệu tham khảo để ghép vào báo cáo chung |

---

## 3. Cách tái lập nhanh

```bash
bash setup.sh                                          # môi trường (Linux + GPU)
bash scripts/move-kd/eval/download_ckpt.sh             # checkpoint → checkpoints/llava-move-kd-7b-v1.1
LIMIT=50 bash scripts/move-kd/eval/pope.sh             # chạy thử
bash scripts/move-kd/eval/{pope,mme,textvqa,sqa}.sh    # chạy full, tổng khoảng 1 giờ 45 phút trên 1 GPU
bash scripts/move-kd/eval/baseline.sh liuhaotian/llava-v1.5-7b
python scripts/move-kd/summarize_results.py --name llava-move-kd-7b-v1.1
python scripts/move-kd/visualize_attention.py          # chạy được trên CPU
```

Dữ liệu eval tải theo [hướng dẫn Evaluation của LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). Các bước chi tiết có trong `Experiments.md` mục 2.7.

## 4. Giới hạn
- Không huấn luyện lại MoVE-KD, nên không tái lập được các ablation (Table 2–5 lấy số từ paper).
- Chỉ chạy 4/8 benchmark. VQAv2, VizWiz và MMBench phải nộp kết quả lên server chấm.
- Mỗi model chỉ chạy một lần, nên các chênh lệch nhỏ (dưới khoảng 0.3 điểm) được coi là dao động.
- Các giải thích về routing ở tầng sâu (token giống nhau hơn) là giả thuyết, nhóm chưa đo trực tiếp.
