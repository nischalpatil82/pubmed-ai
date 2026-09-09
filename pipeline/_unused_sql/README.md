# PubMed literature intelligence — phase 0 / phase 1 starter

Verified against `pubmed26n1443.xml.gz` (23,362 articles, parsed in 15.3s).

## Files

| File | What it does |
|---|---|
| `pubmed_ingest.py` | Streams `.xml.gz` → one CSV per table. Constant memory, parallel across files. |
| `schema.sql` | PostgreSQL DDL sized for ~35M articles / ~250M authorship rows. |
| `demo_queries.sql` | The lead's three questions answered in SQL, plus the entity-resolution check. |

## Run it

```bash
pip install lxml psycopg2-binary

# 1. parse (one file, or a whole directory of them)
python pubmed_ingest.py --src ./pubmed26n1443.xml.gz --out ./csv
python pubmed_ingest.py --src /data/pubmed/baseline  --out ./csv --workers 8

# 2. create the schema
createdb pubmed
psql pubmed -f schema.sql

# 3. bulk load
for t in journals articles authors authorships mesh_headings substances \
         publication_types grants databank_links deleted_pmids; do
  cat csv/${t}__*.csv | psql pubmed -c \
    "COPY ${t} FROM STDIN WITH (FORMAT csv, NULL '')"
done

# 4. answer the questions
psql pubmed -f demo_queries.sql
```

> On a full backfill, drop the indexes in `schema.sql` before the COPY and
> recreate them afterwards. Building them during load is ~10x slower.

## Getting the data

```
ftp.ncbi.nlm.nih.gov/pubmed/baseline/      # ~1,500 files, annual snapshot
ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/   # daily deltas — schedule this
```

Verify every `.md5`. Process `DeleteCitation` blocks (the parser already emits
them to `deleted_pmids`) or retracted papers stay in your index.

Also needed for topic expansion:
`nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml` → `mesh_tree`.

## The one thing to check before trusting KOL output

The `author_key` in this parser is **provisional** — surname + first initial.
On the sample file that gives "Yun Wang" 522 papers across 414 institutions.
That is not one person. The last query in `demo_queries.sql` exposes it.

Entity resolution (phase 2) is the real work:

1. ORCID where present — authoritative, ~34% of distinct authors
2. Blocking + pairwise scoring on affiliation, co-authors, MeSH profile, year proximity
3. Adopt OpenAlex / Semantic Scholar author IDs for the stubborn remainder

## Known gaps in the source data

- **No citation counts** in PubMed XML. Use NIH iCite (free, gives RCR) or OpenAlex.
- **MeSH lags publication** by months, so the newest papers often have no topic codes.
  Fall back to author `Keyword` elements plus a classifier.
- **Abstracts, not full text.** Full text is PMC Open Access — different corpus,
  different licensing.
