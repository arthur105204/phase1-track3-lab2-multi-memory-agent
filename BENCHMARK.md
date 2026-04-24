# BENCHMARK — Lab 17 (10 multi-turn conversations)

## How to run (PowerShell)
```powershell
cd d:\VinAI\phase1-track3-lab2-memory-agent

# With memory
python .\run_benchmark.py --mode mock --out-dir .\outputs\with_memory

# No memory (buffer-only)
python .\run_benchmark.py --mode mock --out-dir .\outputs\no_memory --no-enable-memory
```

Dataset used: `data/bench_conversations.json` (10 conversations).

## Results summary (expected behavior)

| # | Scenario | No-memory result | With-memory result | Pass? (manual) |
|---|----------|------------------|--------------------|-------|
| 1 | Recall preferences (likes Python, dislikes Java) | Forgets or asks again | Uses stored preferences | Check `outputs/*/report.json` |
| 2 | Profile recall (learning ML, knows numpy) | Forgets | Recalls facts from profile | Check `outputs/*/report.json` |
| 3 | Episodic recall (confused async/await) | Generic explanation | Tailors explanation to the remembered confusion | Check `outputs/*/report.json` |
| 4 | Semantic retrieval (LangGraph definition) | Generic / incomplete | Retrieves stored semantic snippet | Check `outputs/*/report.json` |
| 5 | Router mixed intents (pref + episode + fact) | Treats all same | Routes to correct memory type per turn | Check router intent in trace |
| 6 | Trim/token budget stress | May bloat context | Trims lower-priority layers first; policy kept | Check `context_stats.trimmed_layers` |
| 7 | Preference vs policy conflict | Might follow unsafe preference | Policy wins even if preference was stored | Manual read of assistant output |
| 8 | Project fact (Windows + PowerShell) | Generic commands | Suggests PowerShell-appropriate steps | Manual read of assistant output |
| 9 | Conflict update (allergy overwrite) | No stable profile | Updates `allergy` to newest value (đậu nành) | If Redis enabled, check profile hash |
| 10 | End-to-end (pref + fact + semantic + episode) | Weak personalization | Uses all memory types appropriately | Manual read of assistant output |

## Reflection: privacy + limitations (rubric item #5)

### Privacy risks / PII
- **Most sensitive memory**: long-term profile + episodic logs (can contain health data like allergies, or personal preferences tied to identity).
- **Main risks**:
  - Storing sensitive facts without user consent.
  - Wrong retrieval (hallucinated “profile”) can mislead the assistant.
  - Cross-session linking if `user_id` is reused incorrectly.

### Mitigations to mention / implement next
- **Deletion**: support a “forget me” command to delete keys in Redis, remove entries in JSONL episodic log, and delete vectors in Chroma.
- **TTL**: set expiry for preferences/profile keys in Redis to reduce long-term risk.
- **Consent**: ask/confirm before persisting sensitive info (health, finance, etc.).

### Technical limitations (current solution)
- Router + extractor are **heuristic in mock mode**, not robust for real-world language variety.
- Token estimation is **heuristic**, not true tokenizer-based counting.
- Semantic memory depends on Chroma + embedding stack; environment issues may break it without fallback.

