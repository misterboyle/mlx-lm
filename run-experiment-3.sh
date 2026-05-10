#!/usr/bin/env bash
# Experiment 3: Cache growth curve (fixed parsing)

MODEL="/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit"
PORT=30083
LOG=~/.localllm/moe.log
EXTRACT=~/mlx-lm-turbo/extract-tokens.py

get_cache_info() {
  local CACHE=$(grep "Prompt Cache:" "$LOG" | tail -1)
  # Parse: "Prompt Cache: 50 sequences, 12.27 GB"
  local SEQ=$(echo "$CACHE" | sed 's/.*Prompt Cache: \([0-9]*\) sequences.*/\1/')
  local GB=$(echo "$CACHE" | sed 's/.*, \([0-9.]*\) GB/\1/')
  echo "${SEQ}seq/${GB}GB"
}

echo "=== BASELINE ==="
get_cache_info
echo ""

echo "=== EXPERIMENT 3: CACHE GROWTH CURVE ==="
echo ""

call_and_track() {
  local prompt="$1"
  local label="$2"
  
  local RESULT=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
      \"max_tokens\": 10,
      \"temperature\": 0.0
    }" | python3 "$EXTRACT")
  
  local PROMPT_TOKENS=$(echo "$RESULT" | cut -d'|' -f1)
  local CACHE_INFO=$(get_cache_info)
  
  echo "$label: prompt=${PROMPT_TOKENS}tok, cache=${CACHE_INFO}"
}

# Send unique prompts
call_and_track "Hi" "1: tiny (2 chars)"
call_and_track "Hello, how are you today?" "2: very short (25 chars)"
call_and_track "What is the capital of France and what is its population?" "3: short (55 chars)"
call_and_track "Write a detailed explanation of how transformer models work, including attention mechanisms, positional encoding, and feed-forward layers. Be thorough and include technical details." "4: medium (180 chars)"
call_and_track "Explain the complete history of computing from Babbage analytical engine to modern large language models. Cover key milestones, breakthroughs, and paradigm shifts. Include at least 15 distinct historical periods or inventions with dates." "5: long (280 chars)"
call_and_track "Compare and contrast the philosophical approaches of Kant, Hume, and Nietzsche on morality. Discuss their key arguments, differences, and potential syntheses. Include direct quotes from their major works and analyze how their cultural contexts influenced their thinking." "6: very long (350 chars)"
call_and_track "Analyze the economic, social, and political factors that led to the French Revolution. Discuss how these factors interacted and contributed to the outbreak in 1789. Include analysis of the Ancien Regime, the role of the Enlightenment, the financial crisis, and the Estates-General. Evaluate the relative importance of ideological, economic, and political causes." "7: huge (480 chars)"

echo ""
echo "=== EXPERIMENT 3 COMPLETE ==="
echo ""
echo "=== LATEST CACHE STATE ==="
get_cache_info
