#!/usr/bin/env bash
# Experiment 2: Y-value analysis
# Correlate "Prompt processing X/Y" log values with known prompt sizes

MODEL="/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit"
PORT=30083
LOG=~/.localllm/moe.log
EXTRACT=~/mlx-lm-turbo/extract-tokens.py

echo "=== BASELINE ==="
grep "Prompt Cache:" "$LOG" | tail -1
echo ""

echo "=== EXPERIMENT 2: Y-VALUE ANALYSIS ==="
echo ""
echo "The log shows 'Prompt processing progress: X/Y' where Y = tokens NOT in cache"
echo "We send known prompts and correlate Y with prompt_tokens from API"
echo ""

# Clear recent log tail for clean measurement
LOG_START=$(wc -l < "$LOG")

# Send a prompt and capture both API response and log
echo "--- Test 1: Short unique prompt ---"
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"This is a unique test prompt number one for cache analysis.\"}],
    \"max_tokens\": 10,
    \"temperature\": 0.0
  }" | python3 "$EXTRACT"
echo ""
echo "Log entries (Prompt processing):"
grep "Prompt processing" "$LOG" | tail -5
echo ""

# Send the same prompt again
echo "--- Test 2: Same prompt (should hit cache) ---"
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"This is a unique test prompt number one for cache analysis.\"}],
    \"max_tokens\": 10,
    \"temperature\": 0.0
  }" | python3 "$EXTRACT"
echo ""
echo "Log entries (Prompt processing):"
grep "Prompt processing" "$LOG" | tail -5
echo ""

# Send a different short prompt
echo "--- Test 3: Different short prompt ---"
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"This is a different test prompt number two for cache analysis.\"}],
    \"max_tokens\": 10,
    \"temperature\": 0.0
  }" | python3 "$EXTRACT"
echo ""
echo "Log entries (Prompt processing):"
grep "Prompt processing" "$LOG" | tail -5
echo ""

# Send a longer prompt
echo "--- Test 4: Longer unique prompt ---"
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"This is a much longer test prompt that should exercise the cache more significantly. It contains many words and phrases to ensure we have a substantial number of tokens to process and cache. This should help us understand how the cache handles larger prompts and whether there are any patterns in how the Y value relates to the actual prompt size.\"}],
    \"max_tokens\": 10,
    \"temperature\": 0.0
  }" | python3 "$EXTRACT"
echo ""
echo "Log entries (Prompt processing):"
grep "Prompt processing" "$LOG" | tail -5
echo ""

# Send the same long prompt again
echo "--- Test 5: Same long prompt (cache hit) ---"
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"This is a much longer test prompt that should exercise the cache more significantly. It contains many words and phrases to ensure we have a substantial number of tokens to process and cache. This should help us understand how the cache handles larger prompts and whether there are any patterns in how the Y value relates to the actual prompt size.\"}],
    \"max_tokens\": 10,
    \"temperature\": 0.0
  }" | python3 "$EXTRACT"
echo ""
echo "Log entries (Prompt processing):"
grep "Prompt processing" "$LOG" | tail -5
echo ""

echo "=== EXPERIMENT 2 COMPLETE ==="
echo ""
echo "=== LATEST CACHE STATE ==="
grep "Prompt Cache:" "$LOG" | tail -1
