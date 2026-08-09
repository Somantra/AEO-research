# Query Perturbation Study — ChatGPT Brand Positioning

How small wording changes to a search query ("cheapest car insurance for small cars" vs. "car
insurance for small cars") shift the language ChatGPT uses to position a brand in its comparison
tables.

Two consumers who ask nearly the same question can be told different things about the same
brand. That's a new, invisible layer of brand positioning that didn't exist in traditional
search results — a blue link doesn't editorialize; ChatGPT's answer table does. This study
measures that effect directly, using ~4,400 real ChatGPT responses to insurance-shopping queries
in Australia.

## Data

The corpus lives at `../../data/perturbation_chatgpt_tables/` in this repo — one `.txt` file per
ChatGPT search response that contained at least one markdown table:

```
level,query_id,search_request_id,search_result_id,last_updated
<query text>
Tables found: <n>
<markdown table>
End Of Line
...
```

Not every table is a brand comparison — many are generic fact tables (e.g. "common exclusions in
home insurance") with no brand column. `perturbation_chatgpt_brand_position.py` (below) filters
for genuine brand-comparison tables before analyzing anything.

## Why not a standard causal-inference toolkit (e.g. CausalPy)?

Tools like CausalPy (diff-in-diff, regression discontinuity, synthetic control, ...) are built
for *observational* settings, where you infer a treatment effect despite confounders you don't
control. Here the "treatment" is controlled directly — the word added to the query — so the
harder problem isn't confounder-adjustment, it's (a) reliably finding matched query pairs and
(b) separating "the wording change moved the language" from "ChatGPT is non-deterministic and
would have answered differently anyway." Part (b) is handled with a **control group**: when the
corpus contains the exact same query pulled on two different occasions, the language distance
between those two responses is a noise floor — zero wording perturbation, so any language drift
there is pure model noise, not a perturbation effect. Perturbation-driven distances are compared
against that floor instead of relying on statistical confounder adjustment.

## Pipeline

```
data/perturbation_chatgpt_tables/  (this repo)
        │
        ▼
perturbation_chatgpt_brand_position.py
        │  produces perturbation_pairs.csv
        ▼
build_perturbation_brand_positioning_report.py
        │  produces a self-contained report .html
        ▼
open the .html in a browser
```

### Step 0 — install dependencies

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The base pipeline (word-diff + Jaccard distance + the HTML report) needs nothing beyond `scipy`,
which is only used for the optional Kruskal-Wallis significance test — the script runs fine
without it too, just skips that test. **No API key is required for the default run.**

### Step 1 — run the analysis

```bash
python3 perturbation_chatgpt_brand_position.py \
    ../../data/perturbation_chatgpt_tables \
    .
```

Note the console line `Parsed <N> file(s)...` — you'll want that number for `--total-files` in
step 2. For the corpus in this repo, N is 4,445.

Writes `perturbation_pairs.csv` (in the current directory): one row per (brand, base query, perturbed query) pair,
including multi-modifier pairs and any control pairs found, plus a printed summary of median
distance per modifier group and (if `scipy` is installed and there's enough data) a
Kruskal-Wallis test across groups.

Optional flags, both opt-in and both requiring extra setup (`pip install -r
requirements-optional.txt`):
- `--score-sentiment` — adds a sentiment-shift score per pair using a local HuggingFace model
  (`cardiffnlp/twitter-roberta-base-sentiment-latest`, downloaded once from the Hub). No API
  key needed, but pulls in `torch`/`transformers`.
- `--score-embeddings` — adds an embedding-similarity score per pair using the OpenAI API.
  **Requires your own key**: copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or export
  it in your shell). No key ships with this repo, and nothing calls out to OpenAI unless you
  pass this flag.
- `--terms-file` — supply your own list of perturbation terms, one per line, no comments (see
  `terms.example.txt`, which reproduces the built-in default list below as a starting point to
  edit), instead of the built-in default:
  `cheapest`, `cheap`, `most affordable`, `affordable`, `budget`, `best`, `worst`,
  `most reliable`, `least reliable`, `easiest`, `hardest`, `fastest`, `slowest`, `safest`,
  `most trusted`, `most popular`, `premium`.

### Step 2 — build the report

```bash
python3 build_perturbation_brand_positioning_report.py \
    perturbation_pairs.csv \
    report.html \
    --total-files 4445
```

Optional flags:
- `--min-pairs N` (default 4) — minimum number of clean single-modifier pairs a brand needs to
  get a full profile section (lead example + full table). Brands below the threshold land in an
  "Observed but under-sampled" appendix table instead.
- `--total-files` — if omitted, the report's top-line file count falls back to "distinct
  response files referenced" (computed from the pairs themselves), which undercounts since it
  misses files that had no valid pair.

The script only uses **clean single-modifier pairs**: one query with no modifier compared
against the same question with exactly one quality word added. Pairs that carry two stacked
signals (a modifier *and* a follow-up refinement clause) are excluded so each row isolates one
variable.

### Step 3 — read the report

`report.html` is a single self-contained file — no external assets, works offline, open it
directly in a browser. It has:
- A header/stat strip with corpus size and the Kruskal-Wallis significance result.
- A caveat box, stated plainly: **no control-group/noise-floor pairs have been observed in this
  corpus to date** (see Known limitations below) — the report calls this out rather than burying
  it in a methodology footnote.
- Per-brand profiles (left-nav jump list): one card per brand with a lead before/after example
  (biggest word-shift with substantial phrases on both sides), then every modifier tested
  against that brand.
- An appendix for brands seen but below the profile threshold.
- Category comparison matrices (brands × modifiers per product category), so you can compare
  which brands' positioning moves most under a given modifier.

## How the optional scores work

Both optional flags measure the *same thing* the default Jaccard metric does — how much a brand's
positioning language moved between the base query and its perturbed variant — but with richer
signals than surface word overlap. Both operate on a **pair of phrases**: `base_phrase` (how
ChatGPT described the brand *without* the modifier) and `perturbed_phrase` (how it described the
same brand with exactly one quality word, e.g. "cheapest", added). Both also deduplicate first —
every unique phrase is scored once, no matter how many pairs reference it — then attach per-pair
columns to `perturbation_pairs.csv`.

### `--score-sentiment` — did the *tone* move, and which way?

Answers "was the brand described more positively, more negatively, or more neutrally once the
modifier was added?" This is the only score that is **directional**.

- Each phrase runs through a local HuggingFace transformer,
  `cardiffnlp/twitter-roberta-base-sentiment-latest` (a RoBERTa model fine-tuned on tweets to
  classify text as positive / negative / neutral). It runs on your machine (CPU, or CUDA if
  available); the only network use is a one-time model download from the Hub. **No API key, no
  per-call cost.**
- Light preprocessing replaces @-handles and URLs with placeholders — a quirk inherited from the
  model's Twitter training data, so they don't confuse the classifier.
- The model's raw logits are turned into three probabilities that sum to 1.0 via `softmax` (e.g.
  positive 0.70, neutral 0.25, negative 0.05).
- The stored value is a **signed shift**, computed per label as
  `perturbed_probability − base_probability`, giving `sentiment_shift_positive`,
  `sentiment_shift_negative`, and `sentiment_shift_neutral`. A `sentiment_shift_positive` of
  `+0.30` means the perturbed phrasing reads markedly more positive than the base; values near `0`
  mean the tone didn't move.

### `--score-embeddings` — did the *meaning* move?

Answers "is ChatGPT saying roughly the same thing, or something genuinely different?" — semantic
similarity rather than tone.

- Each phrase is sent to the **OpenAI embeddings API** (`text-embedding-3-small` by default,
  override with `OPENAI_EMBEDDING_MODEL`), which returns a high-dimensional vector representing the
  phrase's meaning. **This is the part that requires your own `OPENAI_API_KEY` and costs money per
  call** — small, and kept minimal by the dedup step.
- For each pair it computes **cosine similarity** between the two vectors — the cosine of the angle
  between them, ranging roughly 0→1. `embedding_similarity = 0.95` means the two descriptions mean
  nearly the same thing (wording barely moved the meaning); `0.60` means the meaning shifted
  substantially.
- Note this is a *similarity*, so it runs opposite to the Jaccard *distance*: high similarity =
  small change.

### Why run more than one

The three metrics catch different failure modes of each other. Jaccard can flag that words changed
but not whether it matters; two phrasings can share few words yet mean the same thing (embedding
catches that), and two phrasings can be semantically close yet flip tone (sentiment catches that).
Sentiment is the one that supplies the *directional* narrative the report leads with — "described
as low-cost with the modifier, value-for-money without it" — while embedding tells you whether the
meaning genuinely diverged and Jaccard gives a cheap, always-on baseline.

| | Jaccard (default) | `--score-sentiment` | `--score-embeddings` |
|---|---|---|---|
| Measures | surface word overlap | tone (pos/neg/neutral) | semantic meaning |
| Directional? | no | **yes** — signed shift | no (magnitude only) |
| Cost / setup | free, built-in | free, local model (`torch`) | **OpenAI key, paid** |
| Answers | "did the words change?" | "did the feeling change, and which way?" | "did the meaning change?" |

## Known limitations

- **Pairing is inferred, not designed.** Query pairs are found by normalizing away known
  perturbation terms and matching on the resulting stem, not by deliberately generating a
  base+variant pair for every query up front. On the corpus in this repo, that means most valid
  brand-comparison pairs found are **control pairs** (the identical query pulled on two
  occasions) rather than genuine wording-perturbation pairs — the cleanest fix is a dedicated
  query-pair generator that emits "cheapest X" / "worst X" / ... variants alongside every
  unmodified base query, so every base has full, controlled coverage across the modifier list.
- **Single base sample.** A perturbed variant is only ever compared against one instance of its
  base query. If a base query has been pulled multiple times, averaging its positioning phrase
  (or its embedding) across those pulls before comparing would reduce noise.
- **Brand detection is a header-allowlist + bold heuristic, not a real brand list**
  (`BRAND_COLUMN_HEADERS` in `perturbation_chatgpt_brand_position.py`). It will miss brand
  tables under headers not yet in that list, and won't catch brands ChatGPT doesn't bold. Extend
  the list as new verticals beyond insurance are covered.
- `--score-sentiment` / `--score-embeddings` are opt-in on purpose: sentiment runs a local
  transformer model (slow, no API cost); embeddings hit the OpenAI API (fast, real cost per
  call, deduplicated per unique phrase so each phrase is only embedded once regardless of how
  many pairs reference it).

## Explaining this to a non-technical audience

- **Lead with the concrete example, not the statistic.** "When a shopper adds the word
  'cheapest' to their question, ChatGPT describes Brand X as 'low-cost'; without it, the same
  brand is described as 'value for money'" is legible in one sentence. The Jaccard/embedding
  number is supporting evidence that this happens *systematically*, not the headline.
- **The control-group comparison is the credibility anchor**, once the corpus has genuine
  control pairs to show: "We compared this to what happens when we ask ChatGPT the exact same
  question twice — the language barely moves. It moves much more when we change one word." That
  pre-empts the "isn't this just AI being random?" objection.
- **Actionability**: which perturbation terms move language the most (with enough volume) tells
  a brand which customer-intent variations are worth actively managing vs. which are noise.

## Files in this directory

- `perturbation_chatgpt_brand_position.py` — parses the corpus, finds perturbation pairs,
  computes distance metrics, writes `perturbation_pairs.csv`.
- `build_perturbation_brand_positioning_report.py` — turns `perturbation_pairs.csv` into the
  shareable HTML report.
- `sentiment_scoring.py` / `embedding_scoring.py` — optional scoring modules used by
  `--score-sentiment` / `--score-embeddings` respectively.
- `requirements.txt` / `requirements-optional.txt` — base vs. opt-in dependencies.
- `.env.example` — template for your own `OPENAI_API_KEY`, only needed for
  `--score-embeddings`.
- `terms.example.txt` — starting point for `--terms-file`: the built-in default perturbation
  terms, one per line.
