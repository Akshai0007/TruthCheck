# TruthCheck

A fact-checking agent for AI answers. It reads an answer, pulls out each factual claim, looks for evidence, labels every claim **Supported / Contradicted / Unverified**, and rewrites the answer using only what the evidence backs.

Full project proposal: [docs/TruthCheck_Project_Proposal.docx](docs/TruthCheck_Project_Proposal.docx)

## Try it in 30 seconds (no API key)

```bash
pip install -r requirements.txt
python -m truthcheck.cli --demo          # command line
streamlit run app.py                     # web app: choose "Demo" in the sidebar
python -m pytest -q                      # 40 tests (optional UI/PDF tests may be skipped)
```

Demo mode is a script that only understands one built-in example. It shows the real pipeline, not real fact-checking.

## Use it for real

Pick any provider. Groq and Gemini have free tiers; Ollama runs models on your own computer. Copy the exact model name from the provider's docs.

```bash
export TRUTHCHECK_PROVIDER=groq          # anthropic | openai | groq | gemini | ollama | openrouter
export TRUTHCHECK_MODEL=<model name>
export TRUTHCHECK_API_KEY=<your key>     # not needed for ollama

python -m truthcheck.cli --wikipedia "Marie Curie was born in Paris and won two Nobel Prizes."
python -m truthcheck.cli --docs ./my_faq_folder   # .txt, .md or .pdf files
#  e.g.: "Tuition is due on the 5th of each month."   
python -m truthcheck.cli --ask --samples 3 "Who discovered penicillin?"                      # LLM answers, then we check
```

## How it works

```
answer -> 1 extract claims -> 2 find evidence -> 3 judge each claim -> 4 score -> 5 correct
                                    ^                    |
                                    +-- new query <------+  (if unsure, one more search)
```

## Where each idea came from

These are the repositories we studied. Everything in `truthcheck/` is our own code; the right-hand column is the idea we took.

| Repo | Idea we used | Where |
|---|---|---|
| **VeriScore** (Yixiao-Song) | Extract only *verifiable* claims, written so each stands alone; three labels | `prompts.py` EXTRACT / VERIFY |
| **Factcheck-GPT** (yuxiaw) | Opinions and questions are not check-worthy; split text into passage windows; one verdict + correction per claim | `retrieval.py`, `prompts.py` |
| **SAFE** (google-deepmind) | When unsure, the judge asks for **one new search query** (max steps configurable) | `pipeline.verify_claim` |
| **SelfCheckGPT** (potsawee) | Re-ask the question; ask "is this claim supported by sample N?"; yes=0, no=1, unclear=0.5, averaged | `pipeline._consistency` |
| **OpenFActScore** (lflage) | BM25 passage ranking; skip answers where the model just abstained; score = supported / total | `retrieval.BM25`, `pipeline.check` |
| **FaStFact** (Yingjia-Wan) | Cap the claim count and run claims in parallel to save time | `max_claims`, `workers` |
| **Chain-of-Verification** (paper) | The judge never sees the original answer, only the claim and evidence | `pipeline.verify_claim` |
| **RARR** (paper) | Minimal-edit correction that keeps the original wording and adds source numbers | `pipeline.correct` |

## What we changed on purpose

- **No `eval()` on model output.** Factcheck-GPT runs Python's `eval()` on LLM replies; we parse JSON safely.
- **No trusting the model's memory.** FaStFact accepts quick verdicts from the model's own knowledge; research shows judges misclassify many hallucinations that way. Our judge sees only retrieved evidence.
- **No verdict without a citation.** A "supported" or "contradicted" label that cites no real evidence passage is downgraded to "unverified" (covered by tests).
- **Failures are visible.** If a claim can't be checked (network or model error), it is marked unverified with the error, and a note appears in the report. Nothing is silently dropped.
- **Unchanged if correct.** If every claim is supported, the original answer is returned untouched.

## Measuring accuracy (for your report)

`eval/evaluate.py` scores the verifier on labelled claims (FEVER uses exactly our three labels, as `SUPPORTS / REFUTES / NOT ENOUGH INFO`). Put claims in a `.jsonl` file like `eval/sample_claims.jsonl`, then:

```bash
python -m eval.evaluate my_claims.jsonl --provider groq --model <name> --wikipedia --limit 100
```

It prints accuracy, per-label precision / recall / F1, and a confusion matrix.

### Getting FEVER claims

Download the labelled dev file from [fever.ai](https://fever.ai) (the shared-task `.jsonl`; check the exact file name on the site), then:

```bash
python -m eval.make_fever_sample shared_task_dev.jsonl eval/fever_150.jsonl --per-label 50
```

This writes a balanced sample (50 SUPPORTS, 50 REFUTES, 50 NOT ENOUGH INFO) in the format `evaluate.py` expects.

### Does the second search help?

Run the same claims twice and compare the two result tables:

```bash
python -m eval.evaluate eval/fever_150.jsonl --provider groq --model <n> --wikipedia --max-search-steps 1   # no retry
python -m eval.evaluate eval/fever_150.jsonl --provider groq --model <n> --wikipedia --max-search-steps 2   # default
```

## Cost and speed

One check of an answer with N claims makes about **N + 2 model calls** (1 extract, N verify, 1 correct), up to 2N + 1 if the judge asks for a second search. `--samples K` adds about 2K more. Free tiers have rate limits: lower `workers` in `TruthCheck(...)` if you hit them. Rate-limit replies are retried automatically, wrong keys fail immediately with the provider's message.

## Status: what is and isn't verified

| | Status |
|---|---|
| Pipeline logic, safety guards, scoring, correction | Tested (40 tests; a deliberate break of the evidence guard was caught) |
| Real HTTP code for Anthropic-style, OpenAI-style and Wikipedia-style APIs | Tested against local fake servers: headers, JSON shapes, retries on 429, fast failure on bad keys, unreachable server |
| Random garbage from the model (150 fuzz runs) | Never crashes; always returns a report |
| Non-English text, PDFs, abbreviations like "Dr." | Tested after bugs were found and fixed |
| Clean install with only `requests` | Tested in a fresh virtual environment |
| Web app flow | Click-through tested; screenshot checked for the main view |
| **Live Wikipedia** | **Not tested.** The build sandbox cannot reach Wikipedia. We tested our side of the API shape only. |
| **Real LLM providers** | **Not tested.** No API key was available. Also unconfirmed: whether each model accepts our `temperature` setting and follows the JSON-only instruction reliably. |
| Accuracy numbers | **None yet.** Run the evaluation on 100+ FEVER claims. |

## Known limits (be upfront about these in the proposal)

- Only as good as the evidence source: if Wikipedia or your documents don't cover a claim, it is "unverified", not "true".
- The judge is itself an LLM and can be wrong, or be swayed by instructions hidden in retrieved text. We tell it to treat evidence as data, but this reduces the risk rather than removing it.
- Claims are searched using the claim's own wording. Vague or multi-fact claims may retrieve poor evidence.
- Not suitable for medical, legal or safety decisions.

## Folder map

```
truthcheck/pipeline.py    the agent (extract, verify, consistency, correct, score)
truthcheck/retrieval.py   BM25, local documents, Wikipedia
truthcheck/llm.py         Anthropic + OpenAI-compatible clients (Groq, Gemini, Ollama...)
truthcheck/prompts.py     every instruction given to the model
truthcheck/demo.py        offline scripted demo
app.py                    web interface
eval/evaluate.py          accuracy measurement
eval/make_fever_sample.py turn a FEVER download into our claims file
docs/                     the project proposal
tests/                    40 tests (unit, HTTP-level, fuzz, CLI, UI)
```
