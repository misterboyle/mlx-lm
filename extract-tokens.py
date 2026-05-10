#!/usr/bin/env python3
"""Helper to extract token counts from MoE server responses."""
import sys
import json
import re

text = sys.stdin.read()
# Remove control characters that break JSON parsing
text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
data = json.loads(text)
usage = data.get('usage', {})
print(f"{usage.get('prompt_tokens', '?')}|{usage.get('completion_tokens', '?')}|{usage.get('total_tokens', '?')}")
