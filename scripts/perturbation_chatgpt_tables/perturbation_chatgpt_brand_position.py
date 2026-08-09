# Copyright : Somantra Pty Ltd.
# Licensed under the GNU General Public License v3.0 (see LICENSE / COPYRIGHT).
"""
Query perturbation analysis for ChatGPT brand-positioning language.

Reads extract_chatgpt_tables.py-style output files (see README.md in this directory) from a
local input directory, groups queries into "perturbation families" (same underlying question,
with or without a modifying term like "cheapest"/"best"/"most reliable"), and measures how much
the positioning language ChatGPT uses for each brand shifts between variants.

Full methodology and rationale: README.md in this directory.

Usage:
    python3 perturbation_chatgpt_brand_position.py <input_dir> <output_dir> \\
        [--terms-file TERMS_FILE] [--score-sentiment] [--score-embeddings]

Output:
    <output_dir>/perturbation_pairs.csv, plus a printed summary of per-group median distances
    and (if scipy has enough data) a Kruskal-Wallis test across perturbation-term groups.
"""
import argparse
import csv
import difflib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

DEFAULT_PERTURBATION_TERMS = [
    "cheapest", "cheap", "most affordable", "affordable", "budget",
    "best", "worst", "most reliable", "least reliable",
    "easiest", "hardest", "fastest", "slowest",
    "safest", "most trusted", "most popular", "premium",
]

POSITIONING_COLUMN_HEADERS = {"best for", "ideal for", "good for", "use case", "recommended for"}

# Column-1 headers that indicate the table is actually comparing named brands/companies, as
# opposed to comparing policy types, coverage factors, exclusions, discount types, etc. (which
# also bold their first column for emphasis, but aren't brand tables). Determined empirically:
# across ~5,900 sample files, 'insurer'/'provider' accounted for ~295 bolded-first-column
# tables that were genuine brand comparisons, versus ~500+ bolded tables under headers like
# 'feature'/'factor'/'compare'/'coverage type' that were not. Extend this list as more verticals
# beyond insurance are covered.
BRAND_COLUMN_HEADERS = {
    "insurer", "provider", "company", "brand", "insurance provider", "insurance company",
    "lender", "bank", "retailer",
}

BOLD_CELL_RE = re.compile(r'^\*\*(.+?)\*\*$')
MD_LINK_RE = re.compile(r'\[[^\]]*\]\([^)]*\)')
MD_REF_LINK_RE = re.compile(r'\[[^\]]*\]\[[^\]]*\]')
BOLD_MARKER_RE = re.compile(r'\*\*(.+?)\*\*')
EMPTY_PARENS_RE = re.compile(r'\(\s*\)')


@dataclass
class ParsedFile:
    path: Path
    level: int
    query_id: str
    search_request_id: str
    search_result_id: str
    last_updated: str
    query_text: str
    tables: list = field(default_factory=list)  # list of (header_cells, data_rows)


@dataclass
class PairRecord:
    family_stem: str
    group: str
    direction: str
    terms_added: str
    terms_removed: str
    brand: str
    base_query: str
    perturbed_query: str
    base_search_result_id: str
    perturbed_search_result_id: str
    base_last_updated: str
    perturbed_last_updated: str
    base_phrase: str
    perturbed_phrase: str
    word_diff: str
    jaccard_distance: float
    sentiment_shift_positive: object = ""
    sentiment_shift_negative: object = ""
    sentiment_shift_neutral: object = ""
    embedding_similarity: object = ""


def clean_cell_text(cell: str) -> str:
    """Strip citation links and bold markers, leaving plain positioning language."""
    text = MD_LINK_RE.sub("", cell)
    text = MD_REF_LINK_RE.sub("", text)
    text = BOLD_MARKER_RE.sub(r"\1", text)
    text = EMPTY_PARENS_RE.sub("", text)
    return re.sub(r'\s+', ' ', text).strip()


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def parse_table(table_markdown: str):
    """Return (header_cells, data_rows) for a GFM table, or None if malformed."""
    lines = [l for l in table_markdown.splitlines() if l.strip()]
    if len(lines) < 3:
        return None
    header_cells = split_row(lines[0])
    data_rows = [split_row(l) for l in lines[2:]]
    data_rows = [r for r in data_rows if len(r) == len(header_cells)]
    return header_cells, data_rows


def is_brand_table(header_cells: list[str], data_rows: list[list[str]]) -> bool:
    """True if column 1 is headed like a brand/company column AND most of its data cells are
    bold. Bold-alone isn't enough: ChatGPT also bolds column-1 entries in non-brand comparisons
    (policy types, coverage factors, discount types, ...), so the header check is what filters
    those out."""
    if not data_rows or not header_cells:
        return False
    if header_cells[0].strip().lower() not in BRAND_COLUMN_HEADERS:
        return False
    bold_count = sum(1 for row in data_rows if BOLD_CELL_RE.match(row[0].strip()))
    return bold_count >= max(2, len(data_rows) // 2 + 1)


def positioning_column_index(header_cells: list[str]) -> int | None:
    for i, h in enumerate(header_cells):
        if h.strip().lower() in POSITIONING_COLUMN_HEADERS:
            return i
    return None


def extract_brand_positions(header_cells: list[str], data_rows: list[list[str]]) -> dict[str, str]:
    """brand name -> positioning text, for one brand-comparison table."""
    pos_idx = positioning_column_index(header_cells)
    positions = {}
    for row in data_rows:
        m = BOLD_CELL_RE.match(row[0].strip())
        if not m:
            continue
        brand = clean_cell_text(m.group(1))
        if not brand:
            continue
        if pos_idx is not None:
            phrase = clean_cell_text(row[pos_idx])
        else:
            other_cells = [c for i, c in enumerate(row) if i != 0]
            phrase = clean_cell_text(" | ".join(other_cells))
        if phrase:
            positions[brand] = phrase
    return positions


def parse_file(path: Path) -> ParsedFile | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        return None
    header_fields = lines[0].split(",")
    if len(header_fields) != 5:
        return None
    level, query_id, search_request_id, search_result_id, last_updated = header_fields
    query_text = lines[1]

    tables = []
    current = []
    for line in lines[3:]:
        if line.strip() == "End Of Line":
            if current:
                parsed = parse_table("\n".join(current))
                if parsed:
                    tables.append(parsed)
            current = []
        else:
            current.append(line)

    return ParsedFile(
        path=path, level=int(level), query_id=query_id, search_request_id=search_request_id,
        search_result_id=search_result_id, last_updated=last_updated, query_text=query_text,
        tables=tables,
    )


def load_all(input_dir: Path) -> list[ParsedFile]:
    parsed = []
    for path in sorted(input_dir.glob("*.txt")):
        pf = parse_file(path)
        if pf is not None:
            parsed.append(pf)
    return parsed


def file_brand_positions(pf: ParsedFile) -> dict[str, str]:
    """Merge brand positions across every brand-comparison table in a file (last table wins on
    brand-name collision, which is rare in practice -- one comparison table per response)."""
    merged = {}
    for header_cells, data_rows in pf.tables:
        if is_brand_table(header_cells, data_rows):
            merged.update(extract_brand_positions(header_cells, data_rows))
    return merged


def load_perturbation_terms(terms_file: str | None) -> list[str]:
    if terms_file:
        terms = [l.strip() for l in Path(terms_file).read_text().splitlines() if l.strip()]
    else:
        terms = DEFAULT_PERTURBATION_TERMS
    return sorted({t.lower() for t in terms}, key=len, reverse=True)


def normalize_query(query: str, terms: list[str]) -> tuple[str, list[str]]:
    q = query.lower()
    matched = []
    for term in terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, q):
            matched.append(term)
            q = re.sub(pattern, ' ', q)
    q = re.sub(r'[^\w\s]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q, sorted(matched)


def jaccard_distance(a: str, b: str) -> float:
    tokens_a = set(re.findall(r'\w+', a.lower()))
    tokens_b = set(re.findall(r'\w+', b.lower()))
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    inter = tokens_a & tokens_b
    return 1.0 - (len(inter) / len(union) if union else 0.0)


def word_diff(a: str, b: str) -> str:
    a_words, b_words = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, a_words, b_words)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "delete":
            parts.append("-" + " ".join(a_words[i1:i2]))
        elif tag == "insert":
            parts.append("+" + " ".join(b_words[j1:j2]))
        elif tag == "replace":
            parts.append("-" + " ".join(a_words[i1:i2]))
            parts.append("+" + " ".join(b_words[j1:j2]))
    return " ".join(parts)


def group_label(terms_added: list[str], terms_removed: list[str]) -> tuple[str, str]:
    if not terms_added and not terms_removed:
        return "control_same_query", ""
    if len(terms_added) == 1 and not terms_removed:
        return terms_added[0], "added"
    if len(terms_removed) == 1 and not terms_added:
        return terms_removed[0], "removed"
    return "multi_term", "mixed"


def build_pair_records(parsed_files: list[ParsedFile], terms: list[str]) -> list[PairRecord]:
    families: dict[str, list[tuple[ParsedFile, list[str]]]] = defaultdict(list)
    for pf in parsed_files:
        stem, matched = normalize_query(pf.query_text, terms)
        families[stem].append((pf, matched))

    records = []
    for stem, variants in families.items():
        if len(variants) < 2:
            continue
        variants_sorted = sorted(variants, key=lambda v: (len(v[1]), len(v[0].query_text)))
        base_pf, base_terms = variants_sorted[0]
        base_positions = file_brand_positions(base_pf)
        if not base_positions:
            continue

        for pf, pf_terms in variants_sorted[1:]:
            if pf.search_result_id == base_pf.search_result_id:
                continue
            pf_positions = file_brand_positions(pf)
            if not pf_positions:
                continue

            terms_added = sorted(set(pf_terms) - set(base_terms))
            terms_removed = sorted(set(base_terms) - set(pf_terms))
            same_query_text = pf.query_text.strip().lower() == base_pf.query_text.strip().lower()
            if not same_query_text and not terms_added and not terms_removed:
                # Same normalized stem, different raw text, but no tracked perturbation term
                # differs (e.g. word order or an untracked word changed) -- not something we
                # can attribute, skip rather than mislabel it as a control pair.
                continue

            group, direction = group_label(terms_added, terms_removed)
            common_brands = set(base_positions) & set(pf_positions)
            for brand in sorted(common_brands):
                base_phrase = base_positions[brand]
                pert_phrase = pf_positions[brand]
                if base_phrase == pert_phrase:
                    dist = 0.0
                else:
                    dist = jaccard_distance(base_phrase, pert_phrase)
                records.append(PairRecord(
                    family_stem=stem, group=group, direction=direction,
                    terms_added=",".join(terms_added), terms_removed=",".join(terms_removed),
                    brand=brand,
                    base_query=base_pf.query_text, perturbed_query=pf.query_text,
                    base_search_result_id=base_pf.search_result_id,
                    perturbed_search_result_id=pf.search_result_id,
                    base_last_updated=base_pf.last_updated, perturbed_last_updated=pf.last_updated,
                    base_phrase=base_phrase, perturbed_phrase=pert_phrase,
                    word_diff=word_diff(base_phrase, pert_phrase),
                    jaccard_distance=round(dist, 4),
                ))
    return records


def score_sentiment(records: list[PairRecord]):
    from sentiment_scoring import analyze_sentiment_batch

    unique_texts = sorted({r.base_phrase for r in records} | {r.perturbed_phrase for r in records})
    if not unique_texts:
        return
    print(f"Scoring sentiment for {len(unique_texts)} unique phrase(s)...")
    results = analyze_sentiment_batch(unique_texts)
    scores = {}
    for r in results:
        by_label = {s["label"]: s["score"] for s in r["sentiments"]}
        scores[r["text"]] = (
            by_label.get("positive", 0.0), by_label.get("negative", 0.0), by_label.get("neutral", 0.0),
        )
    for rec in records:
        base_pos, base_neg, base_neu = scores.get(rec.base_phrase, (0.0, 0.0, 0.0))
        pert_pos, pert_neg, pert_neu = scores.get(rec.perturbed_phrase, (0.0, 0.0, 0.0))
        rec.sentiment_shift_positive = round(pert_pos - base_pos, 4)
        rec.sentiment_shift_negative = round(pert_neg - base_neg, 4)
        rec.sentiment_shift_neutral = round(pert_neu - base_neu, 4)


def score_embeddings(records: list[PairRecord]):
    from embedding_scoring import get_embedding, cosine_similarity

    unique_texts = sorted({r.base_phrase for r in records} | {r.perturbed_phrase for r in records})
    print(f"Fetching embeddings for {len(unique_texts)} unique phrase(s)...")
    embeddings = {}
    for text in unique_texts:
        emb = get_embedding(text)
        if emb is not None:
            embeddings[text] = emb
    for rec in records:
        emb_a = embeddings.get(rec.base_phrase)
        emb_b = embeddings.get(rec.perturbed_phrase)
        if emb_a is not None and emb_b is not None:
            rec.embedding_similarity = round(float(cosine_similarity(emb_a, emb_b)), 4)


def write_csv(records: list[PairRecord], out_path: Path):
    fieldnames = [f for f in PairRecord.__dataclass_fields__]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.__dict__)


def print_summary(records: list[PairRecord]):
    by_group = defaultdict(list)
    for r in records:
        by_group[r.group].append(r.jaccard_distance)

    print(f"\n{len(records)} brand-level perturbation pair(s) across {len(by_group)} group(s).\n")
    print(f"{'group':<25}{'n':>6}{'median jaccard distance':>28}")
    rows = []
    for group, dists in by_group.items():
        dists_sorted = sorted(dists)
        median = dists_sorted[len(dists_sorted) // 2]
        rows.append((group, len(dists), median))
    for group, n, median in sorted(rows, key=lambda r: -r[2]):
        marker = "  <- noise floor" if group == "control_same_query" else ""
        print(f"{group:<25}{n:>6}{median:>28.4f}{marker}")

    testable = {g: d for g, d in by_group.items() if len(d) >= 3}
    if scipy_stats is not None and len(testable) >= 2:
        stat, pval = scipy_stats.kruskal(*testable.values())
        print(f"\nKruskal-Wallis across {len(testable)} group(s) with >=3 observations: "
              f"H={stat:.3f}, p={pval:.4f}")
    else:
        print("\nNot enough data yet for a Kruskal-Wallis test (need >=2 groups with >=3 "
              "observations each).")


def main():
    parser = argparse.ArgumentParser(
        description="Measure how ChatGPT's brand-positioning language shifts under query perturbation.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--terms-file", help="One perturbation term per line; overrides the built-in default list.")
    parser.add_argument("--score-sentiment", action="store_true",
                         help="Score sentiment shift per pair via sentiment_scoring.py (local transformer, no API key needed).")
    parser.add_argument("--score-embeddings", action="store_true",
                         help="Score embedding similarity per pair via embedding_scoring.py (calls the OpenAI API -- requires OPENAI_API_KEY).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    terms = load_perturbation_terms(args.terms_file)
    print(f"Using {len(terms)} perturbation term(s): {', '.join(terms)}")

    parsed_files = load_all(input_dir)
    print(f"Parsed {len(parsed_files)} file(s) from {input_dir}.")

    records = build_pair_records(parsed_files, terms)
    if not records:
        print("No perturbation pairs found -- no brand-comparison tables shared a query family.")
        return

    if args.score_sentiment:
        score_sentiment(records)
    if args.score_embeddings:
        score_embeddings(records)

    out_path = output_dir / "perturbation_pairs.csv"
    write_csv(records, out_path)
    print(f"Wrote {len(records)} row(s) to {out_path}")

    print_summary(records)


if __name__ == "__main__":
    sys.exit(main())
