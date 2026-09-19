"""Quick verification of openjev model loading and single-step Tetris inference."""
import time
import torch
from modeling_openjev import OpenJevCrossEncoder, ENT, CON, NEU
from tetris_env import Tetris
from tetris_player import OpenJevPlayer

def main():
    print("[*] Testing openjev cross-encoder inference...")
    t0 = time.time()
    jev = OpenJevCrossEncoder("AlexWortega/openjev", subfolder="qwen3.5-4b-nli", device="cpu", dtype=torch.bfloat16)
    print(f"[+] Loaded openjev in {time.time() - t0:.2f}s")

    # Sanity test
    pairs = [
        ("A Tetris piece lands flat on the bottom. Zero holes are created.", "The move creates no holes."),
        ("A Tetris piece creates 3 covered holes under the stack.", "The move creates no holes.")
    ]
    probs = jev.predict(pairs)
    print("\n--- Sanity Predictions [CON, ENT, NEU] ---")
    for i, (p, h) in enumerate(pairs):
        print(f"P: {p}")
        print(f"H: {h}")
        print(f"-> CON: {probs[i, CON]:.4f} | ENT: {probs[i, ENT]:.4f} | NEU: {probs[i, NEU]:.4f}")

    # Tetris test
    print("\n--- Testing 1-step Tetris Decision ---")
    game = Tetris(seed=42)
    player = OpenJevPlayer(device="cpu")
    # Share loaded encoder
    player.encoder = jev
    move, info = player.choose_move(game)
    print(f"Chosen move (rot, col): {move}")
    print(f"Decision info: {info}")
    print("\n[+] Verification successful!")

if __name__ == "__main__":
    main()
