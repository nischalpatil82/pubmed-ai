-- =====================================================================
-- PubMed literature intelligence - PostgreSQL schema
-- Sized for ~35M articles / ~250M authorship rows (full 2026 baseline).
-- Load path: pubmed_ingest.py -> CSV -> COPY -> staging -> upsert.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- fuzzy name / title matching
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector, for the embedding stage

-- ---------------------------------------------------------------- core

CREATE TABLE journals (
    nlm_id          text PRIMARY KEY,          -- stable key; the title string drifts
    medline_ta      text,                      -- "Stroke"
    title           text,                      -- "Stroke"  (full form)
    iso_abbrev      text,
    issn_linking    text,
    country         text
);
CREATE INDEX journals_ta_trgm ON journals USING gin (medline_ta gin_trgm_ops);

CREATE TABLE articles (
    pmid            bigint PRIMARY KEY,
    nlm_id          text REFERENCES journals(nlm_id),
    title           text,
    abstract        text,
    pub_year        smallint,
    pub_month       smallint,
    volume          text,
    issue           text,
    language        text,
    doi             text,
    pmcid           text,
    status          text,                      -- MEDLINE / PubMed-not-MEDLINE / ...
    date_revised    smallint,                  -- drives idempotent re-ingest
    ingested_at     timestamptz DEFAULT now()
);
CREATE INDEX articles_year        ON articles (pub_year);
CREATE INDEX articles_journal     ON articles (nlm_id);
CREATE INDEX articles_doi         ON articles (doi) WHERE doi IS NOT NULL;

-- Authors as parsed from the XML. author_key is PROVISIONAL:
-- "orcid:0000-...", "name:wang|y", "group:...". Phase 2 replaces it with a
-- resolved person_id - see the entity resolution section below.
CREATE TABLE authors (
    author_key      text PRIMARY KEY,
    display_name    text,
    last_name       text,
    fore_name       text,
    initials        text,
    orcid           text,
    is_group        boolean DEFAULT false,
    person_id       bigint                     -- filled by entity resolution
);
CREATE INDEX authors_orcid   ON authors (orcid) WHERE orcid IS NOT NULL;
CREATE INDEX authors_person  ON authors (person_id);
CREATE INDEX authors_ln_trgm ON authors USING gin (last_name gin_trgm_ops);

CREATE TABLE authorships (
    pmid            bigint,
    author_key      text,
    position        smallint,
    is_first        boolean,
    is_last         boolean,                   -- senior-author signal
    n_authors       smallint,
    affiliation     text,
    institution_id  text,                      -- ROR id, filled in phase 2
    country         text,                      -- parsed from affiliation
    PRIMARY KEY (pmid, position)
);
CREATE INDEX authorships_key     ON authorships (author_key);
CREATE INDEX authorships_country ON authorships (country);

-- ---------------------------------------------------------------- indexing

CREATE TABLE mesh_headings (
    pmid            bigint,
    descriptor_ui   char(7),                   -- D016893
    descriptor_name text,
    is_major        boolean,
    qualifier_uis   text,                      -- pipe-delimited
    PRIMARY KEY (pmid, descriptor_ui)
);
CREATE INDEX mesh_ui ON mesh_headings (descriptor_ui);

-- The MeSH hierarchy, loaded separately from NLM's desc2026.xml.
-- Tree numbers let one query cover a topic AND all its narrower terms.
CREATE TABLE mesh_tree (
    descriptor_ui   char(7),
    tree_number     text,                      -- C14.907.253.535
    descriptor_name text,
    PRIMARY KEY (descriptor_ui, tree_number)
);
CREATE INDEX mesh_tree_prefix ON mesh_tree (tree_number text_pattern_ops);

CREATE TABLE substances (
    pmid            bigint,
    substance_ui    char(7),
    substance_name  text,
    registry_number text,                      -- UNII or CAS; '0' when absent
    PRIMARY KEY (pmid, substance_ui)
);
CREATE INDEX substances_ui   ON substances (substance_ui);
CREATE INDEX substances_name ON substances USING gin (substance_name gin_trgm_ops);

-- Crosswalk built in phase 2 so brand names and classes resolve to one drug.
CREATE TABLE drug_concepts (
    drug_id         bigserial PRIMARY KEY,
    preferred_name  text,
    mesh_ui         char(7),
    rxcui           text,                      -- RxNorm
    unii            text,
    chembl_id       text,
    atc_codes       text[]
);
CREATE TABLE drug_synonyms (
    drug_id         bigint REFERENCES drug_concepts(drug_id),
    synonym         text,
    source          text,
    PRIMARY KEY (drug_id, synonym)
);
CREATE INDEX drug_syn_trgm ON drug_synonyms USING gin (synonym gin_trgm_ops);

CREATE TABLE publication_types (
    pmid            bigint,
    type_ui         char(7),
    type_name       text,
    PRIMARY KEY (pmid, type_ui)
);

CREATE TABLE grants (
    pmid            bigint,
    grant_id        text,
    agency          text,
    country         text
);
CREATE INDEX grants_pmid ON grants (pmid);

-- NCT numbers here are the join key into the ClinicalTrials.gov corpus.
CREATE TABLE databank_links (
    pmid            bigint,
    databank        text,
    accession       text
);
CREATE INDEX databank_acc ON databank_links (databank, accession);

CREATE TABLE deleted_pmids (
    pmid            bigint PRIMARY KEY,
    deleted_at      timestamptz DEFAULT now()
);

-- ---------------------------------------------------------------- external

-- Citation counts do NOT exist in PubMed XML. Loaded from NIH iCite.
CREATE TABLE citation_metrics (
    pmid            bigint PRIMARY KEY,
    citation_count  integer,
    rcr             numeric(8,3),              -- Relative Citation Ratio
    nih_percentile  numeric(5,2),
    refreshed_at    timestamptz
);

-- ---------------------------------------------------------------- embeddings

CREATE TABLE article_embeddings (
    pmid            bigint PRIMARY KEY REFERENCES articles(pmid),
    model           text NOT NULL,
    embedding       vector(768),
    created_at      timestamptz DEFAULT now()
);
-- HNSW after bulk load, never before - building it during ingest is 10x slower.
-- CREATE INDEX ON article_embeddings USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------- views

CREATE MATERIALIZED VIEW article_search AS
SELECT a.pmid, a.title, a.abstract, a.pub_year,
       j.medline_ta AS journal, j.nlm_id,
       to_tsvector('english', coalesce(a.title,'') || ' ' || coalesce(a.abstract,''))
         AS tsv
FROM articles a LEFT JOIN journals j USING (nlm_id);
CREATE INDEX article_search_tsv ON article_search USING gin (tsv);
CREATE UNIQUE INDEX article_search_pmid ON article_search (pmid);
