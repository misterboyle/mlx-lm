#!/usr/bin/env bash
# Run cache profiling experiments
# Uses extract-tokens.py helper for JSON parsing

MODEL="/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit"
PORT=30083
LOG=~/.localllm/moe.log
EXTRACT=~/mlx-lm-turbo/extract-tokens.py

call_api() {
  local prompt="$1"
  curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
      \"max_tokens\": 10,
      \"temperature\": 0.0
    }" | python3 "$EXTRACT"
}

echo "=== BASELINE ==="
grep "Prompt Cache:" "$LOG" | tail -1
echo ""

# Experiment 1: Cache Hit Rate
echo "=== EXPERIMENT 1: CACHE HIT RATE ==="
echo ""

run_prompts() {
  local label="$1"
  shift
  local prompts=("$@")
  
  echo "--- $label ---"
  for p in "${prompts[@]}"; do
    START=$(date +%s%N)
    RESULT=$(call_api "$p")
    END=$(date +%s%N)
    ELAPSED=$(( (END - START) / 1000000 ))
    
    PROMPT_TOKENS=$(echo "$RESULT" | cut -d'|' -f1)
    COMPLETION_TOKENS=$(echo "$RESULT" | cut -d'|' -f2)
    TOTAL_TOKENS=$(echo "$RESULT" | cut -d'|' -f3)
    
    CACHE=$(grep "Prompt Cache:" "$LOG" | tail -1)
    echo "  prompt=${PROMPT_TOKENS}tok, completion=${COMPLETION_TOKENS}tok, total=${TOTAL_TOKENS}tok, time=${ELAPSED}ms"
    sleep 0.3
  done
  echo ""
}

# Tiny prompts
run_prompts "tiny (2-5 chars)" \
  "hi" \
  "hello" \
  "yes" \
  "ok" \
  "bye"

# Short prompts
run_prompts "short (20-40 chars)" \
  "What is the capital of France?" \
  "Tell me a joke." \
  "What is 2+2?" \
  "Define machine learning." \
  "Name three colors."

# Medium prompts
run_prompts "medium (100-200 chars)" \
  "Write a detailed explanation of how transformer models work, including attention mechanisms, positional encoding, and feed-forward layers. Be thorough." \
  "Explain the difference between supervised and unsupervised learning with examples." \
  "Describe the process of photosynthesis in plants, including the light-dependent and light-independent reactions." \
  "What are the main causes of climate change and what are their relative contributions?" \
  "Summarize the key events of World War II in chronological order."

# Long prompts
run_prompts "long (200-300 chars)" \
  "Explain the complete history of computing from Babbage analytical engine to modern large language models. Cover key milestones, breakthroughs, and paradigm shifts. Include at least 15 distinct historical periods or inventions." \
  "Compare and contrast the philosophical approaches of Kant, Hume, and Nietzsche on morality. Discuss their key arguments, differences, and potential syntheses." \
  "Analyze the economic, social, and political factors that led to the French Revolution. Discuss how these factors interacted and contributed to the outbreak in 1789." \
  "Describe the structure and function of the human brain, including the major regions, their roles, and how they interact to produce consciousness and cognition." \
  "Evaluate the pros and cons of nuclear energy as a solution to climate change, considering safety, waste, cost, scalability, and alternatives."

echo "=== EXPERIMENT 1 COMPLETE ==="
echo ""
echo "=== LATEST CACHE STATE ==="
grep "Prompt Cache:" "$LOG" | tail -1
