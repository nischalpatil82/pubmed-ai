"""Request-local evidence state and deterministic, auditable answer rendering."""
import json
import re
from dataclasses import dataclass, field
from time import monotonic


@dataclass
class RequestState:
    snapshot: str
    filters: dict = field(default_factory=dict)
    max_calls: int = 8
    seconds: float = 120
    started: float = field(default_factory=monotonic)
    cache: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    disclosures: list = field(default_factory=list)

    def remaining(self):
        return max(0, self.seconds - (monotonic() - self.started))

    def execute(self, name, arguments, caller):
        key = json.dumps([self.snapshot, self.filters, name, arguments], sort_keys=True)
        if key in self.cache:
            raise ValueError("Repeated action produced no new evidence")
        if not self.remaining() or len(self.calls) >= self.max_calls:
            raise ValueError("Request budget reached")
        result = caller(name, arguments)
        self.calls.append({"tool": name, "args": arguments})
        self.cache[key] = result
        if isinstance(result, dict):
            for field_name in ("caveat", "note", "coverage_note"):
                if result.get(field_name) and result[field_name] not in self.disclosures:
                    self.disclosures.append(result[field_name])
            for row in result.get("results", []):
                if row.get("pmid"):
                    source_texts = [p["text"] for p in row.get("evidence", []) if p.get("section") == "abstract"]
                    if row.get("abstract"):
                        source_texts.append(row["abstract"])
                    if source_texts:
                        self.records[row["pmid"]] = source_texts
        return result


def bounded_payload(value, max_chars=14000):
    """Drop complete rows, never cut JSON or silently cut a source quote."""
    result = json.loads(json.dumps(value, default=str))
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        removed = 0
        while len(json.dumps(result)) > max_chars and result["results"]:
            result["results"].pop()
            removed += 1
        if removed:
            result["omitted_rows"] = removed
    if len(json.dumps(result)) > max_chars:
        return json.dumps({"error": "Tool result exceeds the context budget; request fewer records"})
    return json.dumps(result, default=str)


def render_table(result):
    """Counts and labels are read from typed fields, never rewritten by an LLM."""
    if result.get("error"):
        return "I could not answer that from these files. " + result["error"]
    lines = []
    for key, value in result.items():
        if key in ("results", "note", "caveat", "scope", "dataset", "exact") or value is None:
            continue
        if isinstance(value, (int, float, str)):
            lines.append(f"{key.replace('_', ' ').capitalize()}: {value:,}" if isinstance(value, (int, float))
                         else f"{key.replace('_', ' ').capitalize()}: {value}")
        elif isinstance(value, dict):
            lines.append(f"{key}: " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in value.items()))
    for row in result.get("results", [])[:10]:
        fields = [f"{key.replace('_', ' ')}: {value}" for key, value in row.items()
                  if value is not None and key not in ("abstract", "evidence", "author_key", "affiliation")]
        lines.append("- " + "; ".join(fields))
    if "results" in result and not result["results"]:
        lines.append("No matching records in the selected files.")
    lines.extend(str(result[k]) for k in ("note", "caveat", "coverage_note") if result.get(k))
    return "\n".join(lines)


def render_quotes(proposal, records):
    """Only verbatim, cited evidence is accepted; no semantic accuracy claim."""
    if not isinstance(proposal, dict) or not isinstance(proposal.get("evidence"), list):
        raise ValueError("Expected an evidence list")
    if not 1 <= len(proposal["evidence"]) <= 6:
        raise ValueError("Expected one to six supporting excerpts")
    lines = []
    for item in proposal["evidence"]:
        pmid, quote = str(item.get("pmid", "")), item.get("quote", "")
        if not isinstance(quote, str) or len(quote.strip()) < 20 or len(quote) > 1200:
            raise ValueError("Invalid excerpt length")
        if pmid not in records or not any(quote in source for source in records[pmid]):
            raise ValueError("Citation or excerpt is not supported by retrieved abstract text")
        # Escape HTML and Markdown so source content cannot inject links/UI.
        import html
        safe = html.escape(quote).replace("[", "\\[").replace("]", "\\]").replace("*", "\\*").replace("`", "\\`")
        lines.append(f'> {safe}\n\n[PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)')
    return "Retrieved abstract excerpts (relevance requires review):\n\n" + "\n\n".join(lines)


def render_grounded_claims(proposal, candidates):
    """Render model explanations only when every claim cites supplied excerpts."""
    if not isinstance(proposal, dict) or not isinstance(proposal.get("claims"), list):
        raise ValueError("Expected a claims list")
    if not 1 <= len(proposal["claims"]) <= 5:
        raise ValueError("Expected one to five supported claims")
    by_id = {int(item["id"]): item for item in candidates}
    import html
    blocks = []
    for claim in proposal["claims"]:
        text = claim.get("text", "")
        source_ids = claim.get("source_ids")
        if not isinstance(text, str) or not 20 <= len(text.strip()) <= 700:
            raise ValueError("Invalid claim text")
        if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 3:
            raise ValueError("Every claim needs one to three source IDs")
        try:
            sources = [by_id[int(source_id)] for source_id in source_ids]
        except (KeyError, TypeError, ValueError):
            raise ValueError("Claim references evidence that was not supplied") from None
        safe_text = html.escape(text.strip()).replace("[", "\\[").replace("]", "\\]")
        pmids = []
        quotes = []
        for source in sources:
            pmid, quote = str(source["pmid"]), source["quote"]
            if not isinstance(quote, str) or len(quote.strip()) < 20:
                raise ValueError("Invalid supporting excerpt")
            if pmid not in pmids:
                pmids.append(pmid)
            safe_quote = (html.escape(quote.strip()).replace("[", "\\[")
                          .replace("]", "\\]").replace("*", "\\*")
                          .replace("`", "\\`"))
            quotes.append(f'> {safe_quote}\n\n[PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)')
        citations = ", ".join(f"PMID {pmid}" for pmid in pmids)
        blocks.append(f"{safe_text} ({citations})\n\n" + "\n\n".join(quotes))
    return ("Evidence-grounded summary (review the supporting excerpts):\n\n"
            + "\n\n".join(blocks))


def question_filters(question):
    between = re.search(r"\bbetween (\d{4}) and (\d{4})\b", question, re.I)
    if between:
        return {"since_year": int(between[1]), "until_year": int(between[2])}
    since = re.search(r"\b(?:since|from|after) (\d{4})\b", question, re.I)
    return {"since_year": int(since[1]) + int(since[0].lower().startswith("after"))} if since else {}
