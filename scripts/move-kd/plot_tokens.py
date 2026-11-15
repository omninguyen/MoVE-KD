"""Hình minh hoạ patch token và token [CLS] của CLIP ViT-L/14-336 (dùng cho slide/báo cáo).

    python scripts/move-kd/plot_tokens.py [--image llava/serve/examples/extreme_ironing.jpg]

Ảnh được đệm vuông bằng màu trung bình của CLIP rồi resize 336x336 như expand2square của LLaVA.
Panel thứ ba lấy bản đồ [CLS] attention của CLIP từ images/attn_<tên ảnh>.png (do visualize_attention.py sinh ra).
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

CLIP_MEAN = (122, 116, 104)  # image_mean của CLIP * 255, giống expand2square
SIZE, PATCH = 336, 14
GRID = SIZE // PATCH  # 24
BLUE, ORANGE, GRAY, INK, MUTED = "#0B5FC0", "#C2570C", "#B8C2D3", "#1A2238", "#51607A"


def expand2square(img):
    w, h = img.size
    side = max(w, h)
    out = Image.new("RGB", (side, side), CLIP_MEAN)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out.resize((SIZE, SIZE), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="llava/serve/examples/extreme_ironing.jpg")
    ap.add_argument("--patch", type=int, nargs=2, default=[8, 13], help="hàng, cột của patch tô sáng")
    ap.add_argument("--out", default="images/tokens_cls_vs_patch.png")
    args = ap.parse_args()
    plt.rcParams["font.family"] = ["Calibri", "DejaVu Sans"]

    img = expand2square(Image.open(args.image).convert("RGB"))
    r, c = args.patch
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), gridspec_kw={"width_ratios": [1, 1.25, 1]})

    # (a) ảnh chia lưới 24x24 patch
    ax = axes[0]
    ax.imshow(img)
    for k in range(GRID + 1):
        ax.axhline(k * PATCH - 0.5, color="white", lw=0.4, alpha=0.6)
        ax.axvline(k * PATCH - 0.5, color="white", lw=0.4, alpha=0.6)
    ax.add_patch(Rectangle((c * PATCH - 0.5, r * PATCH - 0.5), PATCH, PATCH, fill=False, ec=ORANGE, lw=3))
    ax.annotate(f"1 ô {PATCH}×{PATCH} px\n→ 1 patch token", xy=(c * PATCH + PATCH, r * PATCH + PATCH / 2),
                xytext=(250, 40), color="white", fontsize=15, fontweight="bold",
                bbox=dict(boxstyle="round", fc=ORANGE, ec="none"), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
    ax.set_title(f"(a) Ảnh {SIZE}×{SIZE} chia thành {GRID}×{GRID} = {GRID * GRID} ô", fontsize=17, color=INK, pad=10)
    ax.axis("off")

    # (b) chuỗi token đưa vào ViT
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(b) Chuỗi token đưa vào ViT: 1 + 576 = 577 token", fontsize=17, color=INK, pad=10)
    xs = [0.3, 1.35, 2.2, 3.05, 4.4, 5.25, 6.6, 7.45, 8.3]
    labels = ["[CLS]", "p1", "p2", "p3", "…", f"p{r * GRID + c + 1}", "…", "p575", "p576"]
    for x, lab in zip(xs, labels):
        if lab == "…":
            ax.text(x + 0.35, 6.9, "…", ha="center", va="center", fontsize=24, color=MUTED)
            continue
        is_cls, is_hl = lab == "[CLS]", lab == f"p{r * GRID + c + 1}"
        fc = BLUE if is_cls else (ORANGE if is_hl else GRAY)
        w = 0.95 if is_cls else 0.7
        ax.add_patch(FancyBboxPatch((x, 6.4), w, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="none"))
        ax.text(x + w / 2, 6.9, lab, ha="center", va="center", fontsize=13 if is_cls else 12,
                color="white" if (is_cls or is_hl) else INK, fontweight="bold")
    # [CLS] chú ý tới các patch
    for x in [1.7, 2.55, 3.4, 5.6, 7.8, 8.65]:
        ax.annotate("", xy=(x, 7.45), xytext=(0.78, 7.45), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2,
                    connectionstyle="arc3,rad=-0.35", alpha=0.75))
    ax.text(4.9, 9.35, "[CLS] chú ý tới mọi patch (self-attention)", ha="center", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(0.3, 5.3, "[CLS]", fontsize=16, color=BLUE, fontweight="bold")
    ax.text(0.3, 4.75, "Token học được, thêm vào\nđầu chuỗi; không ứng với\nvùng ảnh nào. Ở output,\nnó tóm tắt toàn ảnh.",
            fontsize=13.5, color=INK, va="top", linespacing=1.4)
    ax.text(5.35, 5.3, "Patch token (visual token)", fontsize=16, color=ORANGE, fontweight="bold")
    ax.text(5.35, 4.75, "Mỗi token = 1 ô 14×14 của\nảnh, chiếu thành vector\n1024 chiều. LLaVA đưa 576\ntoken này vào LLM.",
            fontsize=13.5, color=INK, va="top", linespacing=1.4)
    ax.text(0.3, 0.4, "Output CLIP ViT-L/14-336: 577 × 1024.\nLLaVA bỏ [CLS], giữ 576 patch token (layer −2).",
            fontsize=12.5, color=MUTED)

    # (c) [CLS] attention -> token weight
    ax = axes[2]
    stem = os.path.splitext(os.path.basename(args.image))[0]
    attn_png = f"images/attn_{stem}.png"
    if os.path.exists(attn_png):
        # ô giữa của hình 3 cột (ảnh | CLIP | student); toạ độ theo bố cục của visualize_attention.py
        ax.imshow(Image.open(attn_png).crop((633, 73, 1167, 607)))
    ax.set_title("(c) Attention của [CLS] tới 576 patch\n= trọng số token W_tok khi chưng cất", fontsize=17, color=INK, pad=10)
    ax.axis("off")

    fig.tight_layout()
    # panel (b) không có ảnh nên cao hơn; kéo về đúng khung của (a) để tiêu đề thẳng hàng
    p0, p1 = axes[0].get_position(), axes[1].get_position()
    axes[1].set_position([p1.x0, p0.y0, p1.width, p0.height])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
