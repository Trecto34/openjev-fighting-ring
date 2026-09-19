"""Fine-tune a small Tetris ranking head on top of Vulkan RoBERTa embeddings.

The transformer stays frozen in llama-server. The trainable head learns to rank legal
placements from a lookahead teacher, which is much better aligned with Tetris play than
the generic NLI labels. Use the resulting directory with ``--policy-head``.
"""
import argparse
import time
from collections import defaultdict

import numpy as np

from tetris_env import Tetris, lookahead_move_scores
from tetris_player import VulkanNLIPlayer, _unique_pairs
from tetris_prompts import generate_candidate_pairs
from modeling_openjev import TetrisRankHead


def collect(args):
    player = VulkanNLIPlayer(url=args.url, model_dir=args.model_dir,
                             strategy=args.strategy, top_k=None)
    pair_rows = []
    target_rows = []
    qid_rows = []
    qid = 0
    t0 = time.time()

    for ep in range(args.episodes):
        env = Tetris(seed=args.seed + ep, max_steps=args.max_steps)
        env.reset()
        while not env.done:
            scored = lookahead_move_scores(env, gamma=args.gamma)
            if not scored:
                break
            legal = [move for move, _ in scored]
            pairs = generate_candidate_pairs(env, legal, strategy=args.strategy)
            unique_pairs, indices = _unique_pairs(pairs)
            # If a prompt strategy ever produces duplicate text for different moves,
            # average their teacher values rather than introducing contradictory labels.
            values = np.asarray([value for _, value in scored], dtype=np.float32)
            grouped = defaultdict(list)
            for pair, idx in zip(pairs, range(len(pairs))):
                grouped[pair].append(float(values[idx]))
            for pair in unique_pairs:
                pair_rows.append(pair)
                target_rows.append(float(np.mean(grouped[pair])))
                qid_rows.append(qid)
            qid += 1
            best_move = max(scored, key=lambda item: item[1])[0]
            env.step(*best_move)
        print(f"episode {ep + 1}/{args.episodes}: pieces={env.pieces_placed} "
              f"lines={env.lines_cleared} decisions={qid}", flush=True)

    if not pair_rows:
        raise RuntimeError("teacher produced no training examples")
    print(f"[*] Embedding {len(pair_rows)} unique move descriptions...", flush=True)
    texts = [f"{premise}</s></s>{hypothesis}" for premise, hypothesis in pair_rows]
    embedded = []
    for start in range(0, len(texts), args.embed_bs):
        stop = min(start + args.embed_bs, len(texts))
        embedded.append(player._embed(texts[start:stop]))
        print(f"  embedded {stop}/{len(texts)}", flush=True)
    X = np.concatenate(embedded, axis=0)
    head = TetrisRankHead(d=X.shape[1], hidden=args.hidden, epochs=args.epochs,
                          patience=args.patience, temperature=args.temperature,
                          seed=args.seed)
    head.fit(X, np.asarray(target_rows, np.float32), np.asarray(qid_rows),
             val_frac=args.val_frac)
    head.save(args.out)
    print(f"[+] saved {args.out} | validation top-1={head.val_acc:.3f} "
          f"| elapsed={time.time() - t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.35,
                        help="next-piece teacher weight")
    parser.add_argument("--strategy", default="qualitative_fine")
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--model-dir", default="/home/server/models/roberta-large-mnli")
    parser.add_argument("--out", default="roberta_tetris_head")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--embed-bs", type=int, default=512,
                        help="texts per embedding HTTP request")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
