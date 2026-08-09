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
- **Tone shift** and **Meaning shift** columns (plus a scores strip on each brand's lead example)
  — but only if the CSV was produced with `--score-sentiment` / `--score-embeddings`. The report
  auto-detects these columns: if they're absent or empty it renders exactly as before, so the base
  pipeline is unchanged. See "How the optional scores work" below for what these columns mean.

## How the word-shift (Jaccard) metric is calculated

Word shift is the default, always-on metric (no flag needed) and the one the report's meter and all
rankings are built on. It is the **Jaccard distance** between the token *sets* of the two
positioning phrases. For a base phrase `a` and perturbed phrase `b` (`jaccard_distance()` in
`perturbation_chatgpt_brand_position.py`):

1. Tokenize case-insensitively into sets — order and repetition discarded:
   `A = set(re.findall(r"\w+", a.lower()))` and `B = set(re.findall(r"\w+", b.lower()))`. The
   pattern `\w+` matches maximal runs of word characters (letters, digits, underscore;
   Unicode-aware in Python's `re`), so punctuation is dropped and serves only as a delimiter.
2. Compute Jaccard distance = 1 − Jaccard similarity:

   ```
   word_shift = 1 − |A ∩ B| / |A ∪ B|
   ```

3. Edge case: if both token sets are empty, the function returns `0.0` (and the `|A ∪ B| = 0`
   division is guarded).
4. The stored `jaccard_distance` column is this value **rounded to 4 decimals**; the report renders
   it to 2 decimals as the "Word shift" meter — `0` = identical token sets, `1` = no shared tokens.

Worked example: `a = "good value for money"` → `A = {good, value, for, money}`;
`b = "the cheapest option, great value"` → `B = {the, cheapest, option, great, value}`. Then
`A ∩ B = {value}` (size 1) and `A ∪ B` has 8 distinct tokens, so `word_shift = 1 − 1/8 = 0.875`.

Properties for review:

- **Set-based, not sequence-based.** Token order and repeated tokens do not affect the score:
  `"value cover"` and `"cover value value"` reduce to the same set. A separate *human-readable,
  ordered* diff is stored in the `word_diff` column, built with `difflib.SequenceMatcher` over the
  whitespace-split word lists (emitting `-removed` / `+added` fragments). That column is display
  annotation only — it is **not** the word-shift number and is not what the meter shows.
- **Range** `[0, 1]`; symmetric in `a` and `b`.
- **No stemming, stop-word removal, or lemmatization** — it is a literal surface-token overlap, so
  two close paraphrases can still score high. This is why the report treats the quoted phrases as
  the primary evidence and the number as supporting context.

## How the optional scores work

Both optional flags measure the *same thing* the default Jaccard metric does — how much a brand's
positioning language moved between the base query and its perturbed variant — but with richer
signals than surface word overlap. When present in `perturbation_pairs.csv`, both are surfaced in
the HTML report automatically (as a **Tone shift** column and a **Meaning shift** column, plus a
scores strip on each brand's lead example); when absent, the report renders exactly as it does for
the base pipeline. Both operate on a **pair of phrases**: `base_phrase` (how
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

#### Exact calculation of the "Tone shift" number

The three per-label shifts above are what's stored in `perturbation_pairs.csv`. The single **Tone
shift** value shown in the HTML report is derived from them as follows. For one pair, let the
sentiment model's probability outputs be:

```
base phrase:      p_base = (p_pos^base, p_neg^base, p_neu^base),   summing to 1
perturbed phrase: p_pert = (p_pos^pert, p_neg^pert, p_neu^pert),   summing to 1
```

The stored per-label shifts are the element-wise difference `Δ = p_pert − p_base`:

```
Δ_positive = p_pos^pert − p_pos^base   (= sentiment_shift_positive)
Δ_negative = p_neg^pert − p_neg^base   (= sentiment_shift_negative)
Δ_neutral  = p_neu^pert − p_neu^base   (= sentiment_shift_neutral)
```

The report's Tone shift is the **net polarity swing**, positive minus negative:

```
tone_shift = Δ_positive − Δ_negative
```

Interpretation and rendering (in `sentiment_net()` / `tone_chip()` in
`build_perturbation_brand_positioning_report.py`):

- `tone_shift > +0.02`  → ▲ "more positive"
- `tone_shift < −0.02`  → ▼ "more negative"
- otherwise             → • "no tone change" (a ±0.02 dead-band to suppress rounding-level noise)

Properties worth noting for review:

- **Range.** Each `Δ ∈ [−1, +1]`, so `tone_shift ∈ [−2, +2]`; in practice it clusters near 0.
- **Why neutral is excluded from the headline.** Because `p_base` and `p_pert` each sum to 1, the
  three shifts sum to zero: `Δ_positive + Δ_negative + Δ_neutral = 0`. So `Δ_neutral` carries no
  independent polarity information — `tone_shift = Δ_positive − Δ_negative` is a pure
  positive-vs-negative contrast, and `Δ_neutral` is reported only in the cell's hover tooltip for
  completeness.
- **Direction, not magnitude of change.** `tone_shift` deliberately does *not* measure how far the
  distribution moved (that would be, e.g., total variation `½·Σ|Δ|`); it measures which way the
  net polarity went. Two pairs with the same `tone_shift` can have moved by different total
  amounts. This is intentional — it's the one metric here meant to answer "which direction," and
  should be read alongside word shift / meaning shift, which carry the magnitude.
- **No control-group normalization.** Like every other metric in this report, `tone_shift` is a raw
  base-vs-perturbed difference and is *not* yet benchmarked against the model's own response
  variance (see Known limitations — no noise-floor pairs exist in the current corpus).

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

#### Exact calculation of the "Meaning shift" number

The `embedding_similarity` column stores the **cosine similarity** between the two phrases'
embedding vectors. For base-phrase embedding `u` and perturbed-phrase embedding `v`
(`cosine_similarity()` in `embedding_scoring.py`):

```
cos_sim = (u · v) / (‖u‖ · ‖v‖)          # dot product ÷ product of L2 norms
```

stored as `embedding_similarity`, **rounded to 4 decimals**. The report's **Meaning shift** value
is derived from it (`meaning_cell()` in `build_perturbation_brand_positioning_report.py`):

```
meaning_shift = max(0, 1 − cos_sim)
```

Properties for review:

- **The `max(0, …)` clamp** guards the theoretical negative-cosine case. OpenAI
  `text-embedding-3-*` vectors are L2-normalized, so `u · v` already *is* the cosine, and in
  practice `cos_sim` sits in roughly `[0.2, 1.0]`, so the clamp rarely binds.
- **Rendering.** Displayed to 2 decimals, with the raw 4-decimal cosine in the cell's hover title;
  `meaning_shift = 0` means identical meaning, larger means more divergent.
- **Not on the same absolute scale as word shift.** Cosine similarity for these embeddings
  compresses into a high, narrow band, so `1 − cos_sim` is systematically smaller than the Jaccard
  word shift for the *same* pair. Compare a pair's meaning shift against *other pairs'* meaning
  shifts, not directly against its own word shift.
- **To reproduce independently:** embed each unique phrase once with the same model (default
  `text-embedding-3-small`, overridable via `OPENAI_EMBEDDING_MODEL`), then apply the two formulas
  above. Embeddings are deduplicated per unique phrase, so identical phrases yield identical vectors
  and therefore `cos_sim = 1`, `meaning_shift = 0`. Note results are only reproducible against the
  same embedding model/version — a different model will shift the absolute numbers.

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
