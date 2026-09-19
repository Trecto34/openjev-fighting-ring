#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PYTHON="$DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    PYTHON="/home/trecto/openjev-tetris/.venv/bin/python"
fi

if [ ! -x "$PYTHON" ]; then
    echo "[!] python-chess venv not found. Install python-chess into a venv and set PYTHON." >&2
    exit 1
fi

cd "$DIR"
MODE="${1:-arena}"
shift || true

case "$MODE" in
    arena|web)
        PORT="${1:-8089}"
        shift 2>/dev/null || true
        echo "[*] Starting AI-vs-AI Chess Arena web viewer on port $PORT..."
        exec "$PYTHON" "$DIR/web_chess.py" --port "$PORT" "$@"
        ;;
    match)
        exec "$PYTHON" "$DIR/chess_arena.py" \
            --white "${WHITE:-minimax}" --black "${BLACK:-heuristic}" \
            --depth "${DEPTH:-3}" "$@"
        ;;
    bench|versus)
        if [ $# -gt 0 ]; then
            EXTRA=("$@")
        else
            EXTRA=(--games 6)
        fi
        exec "$PYTHON" "$DIR/chess_arena.py" --bench \
            --white "${WHITE:-minimax}" --black "${BLACK:-heuristic}" \
            --depth "${DEPTH:-3}" "${EXTRA[@]}"
        ;;
    test)
        echo "[*] Smoke-testing Chess Arena..."
        exec "$PYTHON" -c "
import chess_arena as A
for w, b in [('random','random'), ('heuristic','random'), ('minimax','random')]:
    r = A.run_match(w, b, depth=3, max_plies=40, seed=1)
    print(f'{w:>9} vs {b:<8} ->', r['result']['label'] if r['result'] else 'unfinished')
print('OK: chess_arena import + headless matches pass')
"
        ;;
    *)
        echo "Usage: $0 {arena|web|match|bench|versus|test} [options]"
        echo "  arena [PORT]   start web viewer (default port 8089)"
        echo "  match          headless match (set WHITE/BLACK/DEPTH env)"
        echo "  bench          benchmark series (set WHITE/BLACK/DEPTH/GAMES env)"
        echo "  test           smoke test"
        exit 1
        ;;
esac