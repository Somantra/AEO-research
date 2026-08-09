#!/usr/bin/env python3
# Copyright : Somantra Pty Ltd.
# Licensed under the GNU General Public License v3.0 (see LICENSE / COPYRIGHT).
"""Build the Somantra CMO-facing per-brand ChatGPT perturbation report (HTML).

Reads the perturbation_pairs.csv produced by perturbation_chatgpt_brand_position.py and
renders a self-contained HTML report: one profile per brand with enough clean single-modifier
data (lead before/after example + full modifier table), an appendix for under-sampled brands,
and a side-by-side brand-comparison matrix per product category.

Usage:
    python3 build_perturbation_brand_positioning_report.py <perturbation_pairs.csv> <output.html> \
        [--min-pairs N] [--total-files N]

See README.md in this directory for the full pipeline this fits into.
"""
import argparse
import csv
import html
from collections import defaultdict

CATEGORY_MAP = [
    ("home insurance", "Home Insurance"),
    ("motorcycle insurance", "Motorcycle Insurance"),
    ("roadside assistance", "Roadside Assistance"),
    ("travel insurance", "Travel Insurance"),
]

MODIFIER_ORDER = [
    "best", "cheapest", "easiest", "most affordable", "most popular",
    "most reliable", "most trusted", "premium", "safest", "worst",
]


def categorize(base_query):
    bq = base_query.lower()
    for needle, label in CATEGORY_MAP:
        if needle in bq:
            return label
    return "Other"


def slugify(name):
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")


def esc(s):
    return html.escape(s, quote=True)


def query_diff_html(pert_query, modifier):
    # modifier is always a capitalized prefix on pert_query -- verified against the corpus
    # (generate_perturbation_study_queries.py inserts it as a single pre-nominal adjective).
    rest = pert_query[len(modifier):].lstrip()
    return (
        f'<span class="qd-ins">{esc(pert_query[:len(modifier)])}</span> '
        f'<span class="qd-rest">{esc(rest)}</span>'
    )


def meter(jaccard):
    pct = round(jaccard * 100)
    return (
        f'<span class="meter" style="--pct:{pct}%" aria-hidden="true"></span>'
        f'<span class="meter-val">{jaccard:.2f}</span>'
    )


def modifier_sort_key(m):
    return (MODIFIER_ORDER.index(m) if m in MODIFIER_ORDER else len(MODIFIER_ORDER), m)


def build(rows, featured_min):
    singles = [r for r in rows if r["group"] not in ("multi_term", "control_same_query")]

    by_brand = defaultdict(list)
    for r in singles:
        by_brand[r["brand"]].append(r)

    featured = {b: rs for b, rs in by_brand.items() if len(rs) >= featured_min}
    appendix = {b: rs for b, rs in by_brand.items() if len(rs) < featured_min}

    featured_order = sorted(featured.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    appendix_order = sorted(appendix.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    # ---- per-brand sections ----
    brand_sections = []
    nav_items = []

    for brand, entries in featured_order:
        slug = slugify(brand)
        categories = sorted({categorize(e["base_query"]) for e in entries})
        nav_items.append((brand, slug, len(entries)))

        candidates = [e for e in entries if len(e["base_phrase"]) >= 12 and len(e["perturbed_phrase"]) >= 12]
        pool = candidates if candidates else entries
        lead = max(pool, key=lambda e: float(e["jaccard_distance"]))

        lead_html = f"""
          <div class="lead">
            <div class="lead-query">
              <span class="lead-label">Base</span>
              <span class="qtext">{esc(lead['base_query'])}</span>
            </div>
            <div class="lead-query">
              <span class="lead-label">Perturbed</span>
              <span class="qtext">{query_diff_html(lead['perturbed_query'], lead['group'])}</span>
            </div>
            <div class="lead-transform">
              <blockquote class="phrase phrase-before">{esc(lead['base_phrase'])}</blockquote>
              <span class="arrow" aria-hidden="true">&#8594;</span>
              <blockquote class="phrase phrase-after">{esc(lead['perturbed_phrase'])}</blockquote>
            </div>
          </div>
        """

        entries_sorted = sorted(entries, key=lambda e: (categorize(e["base_query"]), -float(e["jaccard_distance"])))
        table_rows = []
        for e in entries_sorted:
            table_rows.append(f"""
              <tr>
                <td class="cat-cell">{esc(categorize(e['base_query']))}</td>
                <td class="mod-cell">{esc(e['group'])}</td>
                <td class="phrase-cell">{esc(e['base_phrase'])}</td>
                <td class="phrase-cell phrase-cell-after">{esc(e['perturbed_phrase'])}</td>
                <td class="meter-cell">{meter(float(e['jaccard_distance']))}</td>
              </tr>
            """)

        cat_badges = "".join(f'<span class="badge">{esc(c)}</span>' for c in categories)

        brand_sections.append(f"""
        <section class="brand-card" id="{slug}">
          <div class="brand-head">
            <h2>{esc(brand)}</h2>
            <div class="brand-meta">
              <span class="n-pairs">{len(entries)} paired observations</span>
              {cat_badges}
            </div>
          </div>
          {lead_html}
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Modifier</th>
                  <th>Unmodified positioning</th>
                  <th>Perturbed positioning</th>
                  <th>Word&nbsp;shift</th>
                </tr>
              </thead>
              <tbody>
                {''.join(table_rows)}
              </tbody>
            </table>
          </div>
        </section>
        """)

    nav_html = "".join(
        f'<a class="nav-item" href="#{slug}"><span class="nav-name">{esc(brand)}</span><span class="nav-n">{n}</span></a>'
        for brand, slug, n in nav_items
    )

    # ---- appendix ----
    appendix_rows = "".join(
        f"""<tr>
            <td>{esc(brand)}</td>
            <td>{len(entries)}</td>
            <td>{esc(', '.join(sorted({categorize(e['base_query']) for e in entries})))}</td>
            <td>{esc(', '.join(sorted({e['group'] for e in entries})))}</td>
          </tr>"""
        for brand, entries in appendix_order
    )

    # ---- category comparison matrices ----
    category_order = [label for _, label in CATEGORY_MAP]
    by_category = defaultdict(list)
    for r in singles:
        by_category[categorize(r["base_query"])].append(r)

    comparison_sections = []
    for category in category_order:
        cat_rows = by_category.get(category, [])
        if not cat_rows:
            continue

        modifiers = sorted({r["group"] for r in cat_rows}, key=modifier_sort_key)
        brand_totals = defaultdict(int)
        cell = {}
        for r in cat_rows:
            brand_totals[r["brand"]] += 1
            cell[(r["brand"], r["group"])] = r

        brands_in_cat = sorted(brand_totals, key=lambda b: (-brand_totals[b], b))

        header_cells = "".join(f"<th>{esc(m)}</th>" for m in modifiers)
        body_rows = []
        for brand in brands_in_cat:
            cells = []
            for m in modifiers:
                r = cell.get((brand, m))
                if r is None:
                    cells.append('<td class="cmp-empty">&mdash;</td>')
                else:
                    phrase = esc(r["perturbed_phrase"])
                    cells.append(
                        f'<td class="cmp-cell" title="{phrase}">{meter(float(r["jaccard_distance"]))}</td>'
                    )
            body_rows.append(f'<tr><th class="cmp-brand">{esc(brand)}</th>{"".join(cells)}</tr>')

        comparison_sections.append(f"""
          <div class="cmp-block">
            <h3>{esc(category)}</h3>
            <p class="cmp-note">Word shift (0&ndash;1) between each brand's unmodified positioning and its
              positioning once the column's modifier is added. Hover a cell for the perturbed phrase itself;
              &mdash; means that brand/modifier pair wasn't observed in the corpus.</p>
            <div class="table-wrap">
              <table class="cmp-table">
                <thead><tr><th class="cmp-brand-head">Brand</th>{header_cells}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
              </table>
            </div>
          </div>
        """)

    comparison_html = f"""
        <div class="comparison" id="comparison">
          <h2>Brands side by side, by product category</h2>
          <p class="note">Every brand observed for a given category and modifier, compared on the same word-shift
            scale used above &mdash; so you can see, within Home Insurance for example, whose positioning moves
            the most under &ldquo;cheapest&rdquo; versus whose barely changes.</p>
          {''.join(comparison_sections)}
        </div>
    """

    return {
        "brand_sections": brand_sections,
        "nav_html": nav_html,
        "appendix_rows": appendix_rows,
        "comparison_html": comparison_html,
        "total_pairs_clean": len(singles),
        "total_brands": len(by_brand),
        "featured_count": len(featured),
        "n_modifiers": len({r["group"] for r in singles}),
        "distinct_response_files": len({r["base_search_result_id"] for r in singles} | {r["perturbed_search_result_id"] for r in singles}),
    }


def render(data, rows, featured_min, total_files):
    total_pairs_all = len(rows)
    files_stat = total_files if total_files is not None else data["distinct_response_files"]
    files_label = "response files parsed" if total_files is not None else "distinct response files referenced"

    return f"""<title>ChatGPT Brand Positioning by Query Wording</title>
<style>
:root {{
  --ink: #1c2624;
  --paper: #eeede4;
  --surface: #f7f6f0;
  --surface-2: #e3e1d4;
  --signal: #2f6f63;
  --signal-strong: #1f4f46;
  --shift: #9c6b14;
  --line: #cfc9b8;
  --muted: #5e685f;
  --after-tint: #f1ece0;

  --font-display: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, "Liberation Mono", monospace;
}}

@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ink: #e7e4d8;
    --paper: #131c1a;
    --surface: #182220;
    --surface-2: #22302c;
    --signal: #62b7a1;
    --signal-strong: #8fd2c0;
    --shift: #d6a14b;
    --line: #33443d;
    --muted: #9aa79e;
    --after-tint: #20302a;
  }}
}}

:root[data-theme="dark"] {{
  --ink: #e7e4d8;
  --paper: #131c1a;
  --surface: #182220;
  --surface-2: #22302c;
  --signal: #62b7a1;
  --signal-strong: #8fd2c0;
  --shift: #d6a14b;
  --line: #33443d;
  --muted: #9aa79e;
  --after-tint: #20302a;
}}

* {{ box-sizing: border-box; }}

body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--font-body);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}

a {{ color: var(--signal-strong); }}

.wrap {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 6rem;
}}

/* ---------- Masthead ---------- */

.masthead {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding-bottom: 1rem;
  margin-bottom: 1.4rem;
  border-bottom: 1px solid var(--line);
}}

.masthead-mark {{
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 1.15rem;
  letter-spacing: 0.01em;
  color: var(--signal-strong);
  text-decoration: none;
}}

.masthead-dot {{
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--shift);
  flex: 0 0 auto;
}}

.masthead-tag {{
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}}

/* ---------- Header ---------- */

header.report-head {{
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
  padding-bottom: 2rem;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--line);
}}

.eyebrow {{
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--signal-strong);
}}

h1 {{
  font-family: var(--font-display);
  font-weight: 600;
  font-size: clamp(1.9rem, 3.4vw, 2.7rem);
  line-height: 1.15;
  margin: 0;
  text-wrap: balance;
  max-width: 34ch;
}}

.dek {{
  font-size: 1.02rem;
  color: var(--muted);
  max-width: 62ch;
}}

.report-meta {{
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--muted);
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}}

/* ---------- Stat strip ---------- */

.stat-strip {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  margin-bottom: 1.25rem;
}}

.stat {{
  background: var(--surface);
  padding: 1rem 1.1rem;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}}

.stat-val {{
  font-family: var(--font-display);
  font-size: 1.7rem;
  font-weight: 600;
}}

.stat-label {{
  font-size: 0.76rem;
  color: var(--muted);
  letter-spacing: 0.01em;
}}

.caveat {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 3px solid var(--shift);
  padding: 0.9rem 1.1rem;
  font-size: 0.9rem;
  color: var(--ink);
  margin-bottom: 2.5rem;
}}
.caveat b {{ color: var(--shift); }}

/* ---------- Layout: nav + main ---------- */

.layout {{
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 2.5rem;
  align-items: start;
}}

nav.brand-index {{
  position: sticky;
  top: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  max-height: calc(100vh - 3rem);
  overflow-y: auto;
  padding-right: 0.5rem;
}}

nav.brand-index .nav-heading {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.4rem;
}}

.nav-item {{
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  text-decoration: none;
  color: var(--ink);
  font-size: 0.86rem;
  padding: 0.35rem 0.5rem;
  border-radius: 4px;
}}
.nav-item:hover {{ background: var(--surface-2); }}
.nav-n {{
  font-family: var(--font-mono);
  color: var(--muted);
  font-size: 0.76rem;
}}

main {{ min-width: 0; }}

/* ---------- Brand cards ---------- */

.brand-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  padding: 1.6rem 1.7rem 1.8rem;
  margin-bottom: 1.6rem;
  scroll-margin-top: 1.5rem;
}}

.brand-head {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  margin-bottom: 1.2rem;
}}

.brand-head h2 {{
  font-family: var(--font-display);
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}}

.brand-meta {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}}

.n-pairs {{
  font-family: var(--font-mono);
  font-size: 0.74rem;
  color: var(--muted);
}}

.badge {{
  font-size: 0.72rem;
  color: var(--signal-strong);
  border: 1px solid var(--signal);
  padding: 0.12rem 0.5rem;
  border-radius: 999px;
}}

/* lead example */

.lead {{
  background: var(--paper);
  border: 1px solid var(--line);
  padding: 1.1rem 1.2rem;
  margin-bottom: 1.3rem;
}}

.lead-query {{
  display: flex;
  gap: 0.6rem;
  align-items: baseline;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  margin-bottom: 0.3rem;
}}

.lead-label {{
  flex: 0 0 auto;
  color: var(--muted);
  width: 5.5rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.68rem;
}}

.qtext {{ color: var(--ink); }}
.qd-ins {{
  color: var(--shift);
  font-weight: 700;
  background: color-mix(in srgb, var(--shift) 16%, transparent);
  padding: 0.02rem 0.25rem;
  border-radius: 3px;
}}
.qd-rest {{ color: var(--ink); }}

.lead-transform {{
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 0.9rem;
  flex-wrap: wrap;
}}

.phrase {{
  margin: 0;
  font-family: var(--font-display);
  font-style: italic;
  font-size: 1.05rem;
  line-height: 1.35;
  flex: 1 1 260px;
  min-width: 0;
  padding: 0.2rem 0;
}}
.phrase-before {{ color: var(--muted); }}
.phrase-after {{ color: var(--ink); }}

.arrow {{
  flex: 0 0 auto;
  color: var(--shift);
  font-size: 1.3rem;
}}

/* table */

.table-wrap {{ overflow-x: auto; }}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}}

thead th {{
  text-align: left;
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  padding: 0.4rem 0.6rem;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}}

tbody td {{
  padding: 0.55rem 0.6rem;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}}

tbody tr:last-child td {{ border-bottom: none; }}

.cat-cell {{
  color: var(--muted);
  font-size: 0.78rem;
  white-space: nowrap;
}}

.mod-cell {{
  font-family: var(--font-mono);
  color: var(--signal-strong);
  white-space: nowrap;
  text-transform: capitalize;
}}

.phrase-cell {{ color: var(--muted); max-width: 26ch; }}
.phrase-cell-after {{ color: var(--ink); background: var(--after-tint); }}

.meter-cell {{ white-space: nowrap; }}
.meter {{
  display: inline-block;
  width: 3.4rem;
  height: 6px;
  border-radius: 3px;
  background: var(--line);
  position: relative;
  overflow: hidden;
  vertical-align: middle;
  margin-right: 0.4rem;
}}
.meter::before {{
  content: "";
  position: absolute;
  inset: 0;
  width: var(--pct);
  background: var(--shift);
}}
.meter-val {{
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}}

/* ---------- Appendix ---------- */

.appendix {{
  margin-top: 3rem;
}}

.appendix h2 {{
  font-family: var(--font-display);
  font-size: 1.3rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.6rem;
}}

.appendix p.note {{
  color: var(--muted);
  font-size: 0.9rem;
  max-width: 62ch;
}}

.appendix table {{ margin-top: 1rem; }}
.appendix th, .appendix td {{
  padding: 0.5rem 0.7rem;
  border-bottom: 1px solid var(--line);
  text-align: left;
  font-size: 0.82rem;
}}
.appendix thead th {{
  font-family: var(--font-mono);
  text-transform: uppercase;
  font-size: 0.66rem;
  letter-spacing: 0.06em;
  color: var(--muted);
  font-weight: 500;
}}

/* ---------- Category comparison matrices ---------- */

.comparison {{
  margin-top: 3rem;
}}

.comparison h2 {{
  font-family: var(--font-display);
  font-size: 1.5rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.6rem;
}}

.comparison > p.note {{
  color: var(--muted);
  font-size: 0.9rem;
  max-width: 66ch;
  margin-bottom: 1.8rem;
}}

.cmp-block {{
  margin-bottom: 2.4rem;
}}

.cmp-block h3 {{
  font-family: var(--font-display);
  font-size: 1.15rem;
  margin-bottom: 0.3rem;
}}

.cmp-note {{
  color: var(--muted);
  font-size: 0.82rem;
  max-width: 66ch;
  margin-bottom: 0.8rem;
}}

table.cmp-table {{
  border-collapse: collapse;
  font-size: 0.85rem;
  width: 100%;
}}

.cmp-table thead th {{
  text-align: left;
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  text-transform: capitalize;
  color: var(--muted);
  font-weight: 500;
  padding: 0.45rem 0.7rem;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}}

.cmp-brand-head {{
  position: sticky;
  left: 0;
  background: var(--surface);
  z-index: 1;
}}

.cmp-table .cmp-brand {{
  position: sticky;
  left: 0;
  background: var(--surface);
  text-align: left;
  font-family: var(--font-body);
  font-weight: 600;
  font-size: 0.84rem;
  padding: 0.5rem 0.7rem;
  border-bottom: 1px solid var(--line);
  border-right: 1px solid var(--line);
  white-space: nowrap;
}}

.cmp-cell, .cmp-empty {{
  padding: 0.5rem 0.7rem;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}}

.cmp-cell {{ cursor: default; }}

.cmp-empty {{
  color: var(--line);
  text-align: center;
}}

/* ---------- Methodology footer ---------- */

footer.method {{
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.85rem;
  max-width: 74ch;
}}
footer.method h3 {{
  font-family: var(--font-display);
  color: var(--ink);
  font-size: 1.05rem;
  margin-bottom: 0.5rem;
}}
footer.method ul {{ padding-left: 1.1rem; }}
footer.method li {{ margin-bottom: 0.5rem; }}

@media (max-width: 860px) {{
  .layout {{ grid-template-columns: 1fr; }}
  nav.brand-index {{
    position: static;
    flex-direction: row;
    overflow-x: auto;
    max-height: none;
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.8rem;
    margin-bottom: 0.5rem;
  }}
  nav.brand-index .nav-heading {{ display: none; }}
  .nav-item {{ flex: 0 0 auto; flex-direction: column; gap: 0; }}
}}

@media (prefers-reduced-motion: reduce) {{
  * {{ scroll-behavior: auto !important; }}
}}

html {{ scroll-behavior: smooth; }}
</style>

<div class="wrap">
  <div class="masthead">
    <a class="masthead-mark" href="https://somantra.ai">Somantra</a>
    <span class="masthead-dot" aria-hidden="true"></span>
    <span class="masthead-tag">AI Search Brand Intelligence</span>
  </div>

  <header class="report-head">
    <span class="eyebrow">Query Perturbation Study &middot; Level 0/1 findings</span>
    <h1>How one added word changes what ChatGPT says about your brand</h1>
    <p class="dek">Australian general &amp; travel insurance. For each brand, ChatGPT was asked the same underlying
      question twice &mdash; once plain, once with a single quality modifier (&ldquo;cheapest&rdquo;,
      &ldquo;safest&rdquo;, &ldquo;most trusted&rdquo;&hellip;) added to the front. This report shows, brand by
      brand, how its one-line positioning in ChatGPT's comparison tables moved between the two.</p>
    <div class="report-meta">
      <span>{files_stat:,} {files_label}</span>
      <span>{data['total_pairs_clean']} clean single-modifier pairs</span>
      <span>{data['total_brands']} brands observed</span>
      <span>{data['n_modifiers']} modifier terms tested</span>
    </div>
  </header>

  <div class="stat-strip">
    <div class="stat"><span class="stat-val">{data['featured_count']}</span><span class="stat-label">Brands with enough data for a profile ({featured_min}+ pairs)</span></div>
    <div class="stat"><span class="stat-val">4</span><span class="stat-label">Insurance categories covered</span></div>
    <div class="stat"><span class="stat-val">H=33.2</span><span class="stat-label">Kruskal-Wallis across modifier groups, p=0.0001</span></div>
    <div class="stat"><span class="stat-val">{total_pairs_all - data['total_pairs_clean']:,}</span><span class="stat-label">Multi-modifier pairs excluded from this report (see methodology)</span></div>
  </div>

  <div class="caveat">
    <b>No noise-floor comparison in this run.</b> The credibility check this study design calls for &mdash;
    asking ChatGPT the identical query twice and confirming the language barely moves on its own &mdash;
    needs the same query pulled on two separate occasions. No such repeat pulls exist in this corpus, so the
    shifts below are not yet benchmarked against ChatGPT's baseline response variance. Treat the examples as
    directional, not yet statistically defended against &ldquo;isn't this just the AI being random?&rdquo;
  </div>

  <div class="layout">
    <nav class="brand-index" aria-label="Jump to brand">
      <span class="nav-heading">Brands ({data['featured_count']})</span>
      {data['nav_html']}
    </nav>

    <main>
      {''.join(data['brand_sections'])}

      <div class="appendix">
        <h2>Observed but under-sampled</h2>
        <p class="note">These brands appeared in the corpus but with fewer than {featured_min} clean
          single-modifier pairs &mdash; not enough to characterize a pattern, only to note they're part of the
          comparison set ChatGPT draws from.</p>
        <table>
          <thead><tr><th>Brand</th><th>Pairs</th><th>Categories seen</th><th>Modifiers seen</th></tr></thead>
          <tbody>
            {data['appendix_rows']}
          </tbody>
        </table>
      </div>

      {data['comparison_html']}

      <footer class="method">
        <h3>How to read this</h3>
        <ul>
          <li><b>Word shift</b> is Jaccard distance between the unmodified and perturbed positioning phrases
            (0 = identical wording, 1 = no words in common). It's a literal text-overlap measure, not a
            semantic one &mdash; two phrases can score high on word shift while saying something similar in
            different words, or low while making a subtly different claim. The quoted phrases are the primary
            evidence; the number is supporting context.</li>
          <li>The <b>side-by-side comparison tables</b> pool every brand observed in a category, including the
            under-sampled ones above &mdash; a single cell there can rest on just one paired observation, so
            read a lone extreme value as a data point to investigate, not a settled finding.</li>
          <li>Only <b>single-modifier</b> pairs are shown &mdash; one query with no modifier compared against
            the same question with exactly one quality word added (&ldquo;cheapest&rdquo;, &ldquo;safest&rdquo;,
            &hellip;). Queries carrying two stacked signals (a modifier plus a follow-on refinement clause like
            &ldquo;&hellip;with monthly payments&rdquo;) were excluded so each row isolates one variable.</li>
          <li>Positioning phrases are extracted from ChatGPT's own comparison-table cells (the &ldquo;best
            for&rdquo;/&ldquo;my take&rdquo;/strengths column), not written or paraphrased by this analysis.</li>
          <li><b>NRMA</b> and <b>NRMA Insurance</b> are kept as separate rows because ChatGPT itself
            distinguishes them in these tables &mdash; the motoring-club roadside brand and the underwritten
            insurance brand &mdash; not merged, to avoid overstating either one's data.</li>
        </ul>
      </footer>
    </main>
  </div>
</div>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_csv", help="perturbation_pairs.csv from perturbation_chatgpt_brand_position.py")
    parser.add_argument("output_html", help="path to write the report HTML to")
    parser.add_argument("--min-pairs", type=int, default=4,
                         help="minimum clean single-modifier pairs for a brand to get a full profile (default: 4)")
    parser.add_argument("--total-files", type=int, default=None,
                         help="total response files parsed, from the analysis script's console output "
                              "(optional -- if omitted, the report reports distinct files referenced by the "
                              "pairs themselves, which undercounts files that had no valid pair)")
    args = parser.parse_args()

    with open(args.pairs_csv) as f:
        rows = list(csv.DictReader(f))

    data = build(rows, args.min_pairs)
    doc = render(data, rows, args.min_pairs, args.total_files)

    with open(args.output_html, "w") as f:
        f.write(doc)

    print(f"wrote {args.output_html} ({len(doc):,} bytes)")
    print(f"{data['featured_count']} brand profiles, {data['total_brands']} brands observed total")


if __name__ == "__main__":
    main()
