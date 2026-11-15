"""So sánh MoVE-KD với baseline LLaVA-1.5 cho video demo.

    # 1) câu POPE mà hai model trả lời khác nhau (dùng file answers đã chạy full)
    python scripts/move-kd/demo_compare.py pope --a llava-move-kd-7b-v1.1 --b llava-v1.5-7b [--n 8]
    # 2) đặt cạnh nhau câu trả lời của demo_ask.py cho cùng bộ câu hỏi
    python scripts/move-kd/demo_compare.py ask demo_out/llava-move-kd-7b-v1.1.json demo_out/llava-v1.5-7b.json
"""
import argparse
import json
import os
import textwrap

POPE = "playground/data/eval/pope"


def yes(text):
    # cùng quy tắc với llava/eval/eval_pope.py: câu đầu có No/no/not -> "no", còn lại -> "yes"
    words = text.split(".")[0].replace(",", "").split(" ")
    return not ("No" in words or "not" in words or "no" in words)


def pope(args):
    labels = {}
    for split in ("adversarial", "popular", "random"):
        for line in open(f"{POPE}/coco/coco_pope_{split}.json"):
            q = json.loads(line)
            labels.setdefault(split, []).append(q["label"] == "yes")
    qs = [json.loads(l) for l in open(f"{POPE}/llava_pope_test.jsonl")]
    load = lambda tag: {d["question_id"]: d["text"] for d in map(json.loads, open(f"{POPE}/answers/{tag}.jsonl"))}
    A, B = load(args.a), load(args.b)
    seen, rows = {}, []
    for q in qs:  # thứ tự câu trong mỗi split khớp thứ tự nhãn, như eval_pope.py
        i = seen.get(q["category"], 0)
        seen[q["category"]] = i + 1
        gt = labels[q["category"]][i]
        if q["question_id"] in A and q["question_id"] in B:
            ra, rb = yes(A[q["question_id"]]) == gt, yes(B[q["question_id"]]) == gt
            if ra != rb:
                rows.append((ra, q, gt, A[q["question_id"]], B[q["question_id"]]))
    win = sum(r[0] for r in rows)
    print(f"Hai model trả lời khác nhau ở {len(rows)} câu: {args.a} đúng {win}, {args.b} đúng {len(rows) - win}\n")
    for want in (True, False):
        print(f"--- {args.a} {'ĐÚNG' if want else 'SAI'}, {args.b} {'SAI' if want else 'ĐÚNG'} ---")
        for ra, q, gt, a, b in [r for r in rows if r[0] == want][: args.n]:
            print(f"  [{q['category']}] {q['image']}: {q['text'].splitlines()[0]}  (nhãn: {'yes' if gt else 'no'})")
            print(f"      {args.a}: {a:<5}  {args.b}: {b}")
        print()


def ask(args):
    a, b = (json.load(open(p)) for p in (args.a_file, args.b_file))
    for ra, rb in zip(a["results"], b["results"]):
        print("=" * 100)
        print(f"{os.path.basename(ra['image'])}: {ra['question']}")
        for tag, r in ((a["model"], ra), (b["model"], rb)):
            print(f"\n  [{tag}]")
            print(textwrap.indent(textwrap.fill(r["answer"], 94), "    "))
        print()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pope")
    p.add_argument("--a", default="llava-move-kd-7b-v1.1")
    p.add_argument("--b", default="llava-v1.5-7b")
    p.add_argument("--n", type=int, default=6)
    p.set_defaults(fn=pope)
    q = sub.add_parser("ask")
    q.add_argument("a_file")
    q.add_argument("b_file")
    q.set_defaults(fn=ask)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
