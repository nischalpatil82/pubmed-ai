-- =====================================================================
-- The three questions the lead asked, answered in SQL.
-- Verified against pubmed26n1443 (23,362 articles) - real output in comments.
-- =====================================================================

-- ---------------------------------------------------------------------
-- Q1  "List me the journals publishing on <topic>"
--     Exhaustive and counted. Vector search cannot do this.
-- ---------------------------------------------------------------------
SELECT j.medline_ta                AS journal,
       j.country,
       count(DISTINCT a.pmid)      AS papers
FROM   substances s
JOIN   articles  a ON a.pmid   = s.pmid
JOIN   journals  j ON j.nlm_id = a.nlm_id
WHERE  s.substance_name = 'Antineoplastic Agents'
GROUP  BY 1, 2
ORDER  BY papers DESC;
--  Int J Mol Sci              Switzerland   16
--  Support Care Cancer        Germany       10
--  Anticancer Agents Med Chem Netherlands   10
--  ... 99 journals in total, from ONE file.


-- ---------------------------------------------------------------------
-- Q2  "List me the drugs"
--     Already curated by NLM, with UNII/CAS registry numbers attached.
-- ---------------------------------------------------------------------
WITH hits AS (
    SELECT pmid FROM substances WHERE substance_name = 'Antineoplastic Agents'
)
SELECT s.substance_name  AS drug,
       s.registry_number AS unii_or_cas,
       count(*)          AS papers
FROM   substances s
JOIN   hits h ON h.pmid = s.pmid
WHERE  s.substance_name <> 'Antineoplastic Agents'
  AND  s.registry_number <> '0'
GROUP  BY 1, 2
ORDER  BY papers DESC;
--  Cisplatin     Q20Q21Q62J   13
--  Doxorubicin   80168379AG    4
--  Carboplatin   BG3F62OND5    4


-- ---------------------------------------------------------------------
-- Q3  "Top key opinion leaders"
--     A defined, defensible score - not a vibe.
--     Tune the weights WITH the client, then freeze them.
-- ---------------------------------------------------------------------
WITH hits AS (
    SELECT DISTINCT pmid FROM substances WHERE substance_name = 'Antineoplastic Agents'
),
scored AS (
    SELECT au.author_key,
           max(a.display_name)                        AS name,
           max(a.orcid)                               AS orcid,
           count(DISTINCT au.pmid)                    AS papers,
           count(*) FILTER (WHERE au.is_first)        AS first_author,
           count(*) FILTER (WHERE au.is_last)         AS senior_author,
           max(ar.pub_year)                           AS latest_year,
           -- score: volume + authorship position + recency
           count(DISTINCT au.pmid) * 1.0
             + count(*) FILTER (WHERE au.is_first) * 1.5
             + count(*) FILTER (WHERE au.is_last)  * 2.0
             + count(*) FILTER (WHERE ar.pub_year >= 2025) * 1.0
             AS kol_score
    FROM   authorships au
    JOIN   hits     h  ON h.pmid  = au.pmid
    JOIN   articles ar ON ar.pmid = au.pmid
    JOIN   authors  a  ON a.author_key = au.author_key
    WHERE  NOT a.is_group
    GROUP  BY au.author_key
)
SELECT name, orcid, papers, first_author, senior_author, latest_year,
       round(kol_score, 1) AS kol_score
FROM   scored
ORDER  BY kol_score DESC
LIMIT  20;


-- =====================================================================
-- THE CATCH - run this before trusting Q3.
-- =====================================================================
-- A naive author key (surname + first initial) silently merges different
-- people. This query exposes it. Real output from the sample file:
--
--   display_name    papers   distinct_institutions
--   Yun Wang           522                     414
--   Lei Wang           228                     199
--   Lei Zhang          183                     161
--
-- 522 papers across 414 institutions is not one researcher. It is dozens.
-- Entity resolution (phase 2) is what makes the KOL numbers real.
SELECT a.display_name,
       count(DISTINCT au.pmid)                                  AS papers,
       count(DISTINCT split_part(au.affiliation, ',', 1))       AS distinct_institutions
FROM   authorships au
JOIN   authors a USING (author_key)
WHERE  NOT a.is_group
GROUP  BY 1
HAVING count(DISTINCT split_part(au.affiliation, ',', 1)) > 50
ORDER  BY papers DESC
LIMIT  20;


-- ---------------------------------------------------------------------
-- Topic expansion: match a MeSH term AND everything below it in the tree.
-- Without this, a query for "Carotid Stenosis" misses its narrower terms.
-- ---------------------------------------------------------------------
WITH topic AS (
    SELECT tree_number FROM mesh_tree WHERE descriptor_ui = 'D016893'
),
descendants AS (
    SELECT DISTINCT mt.descriptor_ui
    FROM   mesh_tree mt, topic t
    WHERE  mt.tree_number LIKE t.tree_number || '%'
)
SELECT count(DISTINCT mh.pmid) AS papers_in_topic_and_below
FROM   mesh_headings mh
JOIN   descendants d USING (descriptor_ui);
