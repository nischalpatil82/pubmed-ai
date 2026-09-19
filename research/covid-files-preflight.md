# covid-files dataset preflight

Use only `C:/Users/User/Downloads/pubmed25n1509 (1).zip` for this dataset. The
earlier 20-file dataset in the repository's `store/` and `index/` directories
must remain separate. The branch name does not establish that the source
records are exclusively about COVID; no topic filter has been requested.

## Archive inspection

- Archive size: 9,888,241,393 bytes.
- XML members: 159.
- Uncompressed XML size: 62,065,088,769 bytes.
- ZIP directory: readable.
- Three sample members passed complete XML parsing and ZIP CRC validation.
- Integrity of all 159 members has not yet been checked.

| Sample | Article occurrences | With abstract | Deletion events |
|---|---:|---:|---:|
| pubmed25n1509.xml | 14,664 | 13,230 | 234 |
| pubmed25n1284.xml | 13,046 | 11,580 | 38 |
| pubmed25n1505.xml | 20,987 | 18,926 | 20 |

Extrapolating abstract occurrences by XML byte size from these three samples
gives approximately 3.22 million embedding inputs before cross-file
deduplication, or approximately 4.94 GB of raw 384-dimensional float32 vectors.
This is a rough sample estimate, not a unique-article count or a total build
size. The fact store, keyword index, working CSVs and temporary files require
additional space. Revisions and deletions need ordered reconciliation.

## Storage and build status

C: had approximately 3.1 GB free during inspection on 10 September 2026. No
other filesystem drive was available. Full ingestion and embedding have not
started. A separate output location with more free space is needed. Roughly
40 GB of free working space is a provisional planning allowance for a streamed
build; actual requirements must be checked during ingestion.

The importer now accepts ZIP archives directly, as well as XML and gzipped
XML. It reads members without extracting 62.1 GB of XML, orders source names,
and rejects duplicate member stems that would overwrite output CSV files.

After selecting an output location, use a dedicated `covid-files` dataset
directory for all intermediate CSVs, the fact store and search indexes. Set
`PUBMED_STORE` and `PUBMED_INDEX` to that dataset's folders for every build and
serving command. The existing root launch scripts still point to the earlier
dataset and must not be used for this build without updating their configuration.

No embedding model has been trained or new index built yet. No branch or model
artifacts have been pushed. ZIP, XML, intermediate files and indexes should
remain outside ordinary Git commits; completed artifact publishing needs its
own destination distinct from the previous dataset.
