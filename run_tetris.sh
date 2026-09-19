#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PYTHON="$DIR/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "[!] Virtual environment not found at $DIR/.venv. Please run setup first."
    exit 1
fi

cd "$DIR"

MODE="${1:-zero-shot}"
shift || true

case "$MODE" in
    zero-shot|play)
        echo "[*] Running openjev Zero-Shot Tetris Player..."
        "$PYTHON" "$DIR/tetris_player.py" --mode zero-shot "$@"
        ;;
    oracle)
        echo "[*] Running Dellacherie Oracle Player..."
        "$PYTHON" "$DIR/tetris_player.py" --mode oracle "$@"
        ;;
    random)
        echo "[*] Running Random Baseline..."
        "$PYTHON" "$DIR/tetris_player.py" --mode random "$@"
        ;;
    train-mlp)
        echo "[*] Training LatentMLPHead with openjev latents..."
        "$PYTHON" "$DIR/tetris_player.py" --mode train-mlp "$@"
        ;;
    coop)
        PORT="${1:-8088}"
        shift || true
        echo "[*] Starting openjev Coop Tetris (Human + openjev) on port $PORT..."
        "$PYTHON" "$DIR/web_coop.py" --port "$PORT" "$@"
        ;;
    versus|vulkan|all)
        # Start exactly the llama-servers the requested flags need.
        # "all" and "versus" include Heuristic + SalesRL + Vulkan + Qwen.
        PORT="${1:-8088}"
        shift || true
        if [ "$MODE" = "all" ] || [ "$MODE" = "versus" ]; then
            ARGS="--with-heuristic --with-kev --with-laya --with-spark --free-run $*"
        else
            ARGS="--with-vulkan $*"
        fi
        LLAMA="${LLAMA_SERVER:-/home/server/llama.cpp/build/bin/llama-server}"

        start_srv() {  # $1=port $2=gguf $3.. extra llama-server args
            curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && \
                { echo "[*] llama-server already up on :$1"; return; }
            local port="$1" gguf="$2"; shift 2
            if [ ! -f "$gguf" ]; then echo "[!] missing model: $gguf" >&2; exit 1; fi
            echo "[*] Starting llama-server (Vulkan) on :$port ..."
            # -np 32 matters: the server gives each sequence its own slot, so with the
            # default 4 a 30-move batch runs as 8 sequential rounds (465ms -> 287ms).
            "$LLAMA" -m "$gguf" --port "$port" --host 127.0.0.1 -ngl 99 \
                -np 32 -b 2048 -ub 2048 -c 16384 --cont-batching --flash-attn on \
                --cache-prompt "$@" >"/tmp/llama-$port.log" 2>&1 &
            for _ in $(seq 1 90); do
                curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return
                sleep 1
            done
            echo "[!] llama-server on :$port did not come up, see /tmp/llama-$port.log" >&2
            exit 1
        }

        case "$ARGS" in *--with-vulkan*)
            start_srv 8091 "${NLI_GGUF:-/home/server/models/roberta-large-mnli-f16.gguf}" \
                --embeddings --pooling cls --embd-normalize -1 ;;
        esac
        case "$ARGS" in *--with-qwen*)
            start_srv 8092 "${QWEN_GGUF:-/home/server/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}" ;;
        esac
        case "$ARGS" in *--with-openjev*)
            echo "[!] openjev runs on CPU at ~21s/piece. Lock-step holds every board to the"
            echo "    slowest one, so all boards will step at that pace; add --free-run to unlock." ;;
        esac

        echo "[*] Starting Coop Tetris on port $PORT ..."
        "$PYTHON" "$DIR/web_coop.py" --port "$PORT" $ARGS
        ;;
    sales-rl|sales)
        echo "[*] Running SalesRLAgent (PPO Conversion Model from arXiv:2503.23303)..."
        "$PYTHON" "$DIR/sales_rl_player.py" "$@"
        ;;
    kev)
        echo "[*] Running kev-0.5b Decision Model Player (jaredpalmer/kev-0.5b)..."
        "$PYTHON" "$DIR/kev_player.py" "$@"
        ;;
    laya)
        echo "[*] Running Laya RLCD Decision Model Player (convaiinnovations/laya)..."
        "$PYTHON" "$DIR/laya_player.py" "$@"
        ;;
    spark)
        echo "[*] Running djev-spark DiffusionGemma System 1 Player..."
        "$PYTHON" "$DIR/tetris_player.py" --mode spark "$@"
        ;;
    web)
        PORT="${1:-8088}"
        shift || true
        echo "[*] Starting openjev Coop Tetris Web Dashboard on port $PORT..."
        "$PYTHON" "$DIR/web_coop.py" --port "$PORT" "$@"
        ;;
    test)
        echo "[*] Testing Tetris Environment..."
        "$PYTHON" -c "from tetris_env import Tetris, oracle_policy; t = Tetris(); print(t.render_ascii())"
        ;;
    *)
        echo "Usage: $0 {all|versus|kev|laya|sales-rl|vulkan|coop|web|play|oracle|random|train-mlp|test} [options]"
        exit 1
        ;;
esac
