"""PubMed API service — fetch research abstracts and full-text articles.

Features:
  - Structured per-article results (title, PMID, abstract, full-text)
  - PMID → PMCID conversion via NCBI ID Converter API
  - PMC full-text retrieval via efetch
  - Europe PMC REST API fallback for full-text
  - Section-based intelligent chunking for full-text articles
  - Graceful fallback: full-text → abstract
  - Rate limiting (≤3 req/sec without API key)
"""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Updated 2025: NCBI permanently moved the ID Converter API to a new domain
NCBI_CONVERTER = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


@dataclass
class PubMedArticle:
    """A single PubMed article with metadata and content."""
    pmid: str
    title: str = ""
    abstract: str = ""
    full_text: str = ""
    pmcid: str | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    pub_year: str = ""
    sections: dict[str, str] = field(default_factory=dict)  # section_name -> content
    full_text_source: str = ""  # "pmc" | "europepmc" | ""

    @property
    def has_full_text(self) -> bool:
        return bool(self.full_text.strip())

    @property
    def best_content(self) -> str:
        """Return the best available content (full-text preferred)."""
        return self.full_text if self.has_full_text else self.abstract

    @property
    def display_name(self) -> str:
        """Human-readable source name for source_document field."""
        name = self.title[:80] if self.title else "Untitled"
        if self.pmid:
            name += f" - PMID:{self.pmid}"
        return name


@dataclass
class PubMedResult:
    """Aggregated result of a PubMed fetch operation."""
    abstracts: list[str]  # backward-compatible flat list
    articles: list[PubMedArticle] = field(default_factory=list)
    total_ids_found: int = 0
    total_fetched: int = 0
    full_text_count: int = 0
    abstract_only_count: int = 0
    errors: list[str] = field(default_factory=list)


# ── ID Conversion: PMID → PMCID ───────────────────────────────────


async def _convert_pmids_to_pmcids(
    client: httpx.AsyncClient,
    pmids: list[str],
    email: str,
    rate_limit_delay: float,
) -> dict[str, str]:
    """Convert PMIDs to PMCIDs using NCBI ID Converter API.

    Returns a dict mapping PMID -> PMCID for articles that have PMC records.
    Tries the new PMC v1 API first, falls back to the legacy NCBI converter.
    """
    mapping: dict[str, str] = {}
    if not pmids:
        return mapping

    batch_size = 200  # NCBI converter limit
    batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]

    for batch_idx, batch in enumerate(batches):
        try:
            await asyncio.sleep(rate_limit_delay)
            # Primary: new PMC ID Converter v1 API
            resp = await client.get(
                NCBI_CONVERTER,
                params={
                    "ids": ",".join(batch),
                    "idtype": "pmid",
                    "format": "json",
                    "tool": "medqa-dataset-generator",
                    "email": email,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            batch_found = 0
            for record in data.get("records", []):
                pmid = str(record.get("pmid", ""))
                pmcid = str(record.get("pmcid", ""))
                if pmid and pmcid:
                    mapping[pmid] = pmcid
                    batch_found += 1
            if batch_found > 0:
                logger.info("PMID→PMCID batch %d: %d/%d converted", batch_idx + 1, batch_found, len(batch))
                continue

            # If primary returned 0, try legacy endpoint as fallback
            logger.info("Primary converter returned 0 — trying legacy endpoint for batch %d", batch_idx + 1)
            await asyncio.sleep(rate_limit_delay)
            legacy_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            resp2 = await client.get(
                legacy_url,
                params={
                    "ids": ",".join(batch),
                    "idtype": "pmid",
                    "format": "json",
                    "tool": "medqa-dataset-generator",
                    "email": email,
                },
                timeout=30.0,
            )
            resp2.raise_for_status()
            data2 = resp2.json()
            for record in data2.get("records", []):
                pmid = str(record.get("pmid", ""))
                pmcid = str(record.get("pmcid", ""))
                if pmid and pmcid:
                    mapping[pmid] = pmcid

        except Exception as e:
            logger.warning("PMID→PMCID conversion batch %d failed: %s", batch_idx + 1, e)

    logger.info("PMID→PMCID: %d of %d have PMC records", len(mapping), len(pmids))
    return mapping


# ── Structured Abstract Fetch ──────────────────────────────────────


async def _fetch_abstracts_xml(
    client: httpx.AsyncClient,
    pmids: list[str],
    email: str,
    rate_limit_delay: float,
    batch_size: int = 100,
) -> list[PubMedArticle]:
    """Fetch abstracts in XML format with full metadata."""
    articles: list[PubMedArticle] = []
    batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]

    for i, batch in enumerate(batches):
        try:
            await asyncio.sleep(rate_limit_delay)
            resp = await client.get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "xml",
                    "retmode": "xml",
                    "email": email,
                },
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            for article_el in root.findall(".//PubmedArticle"):
                pmid_el = article_el.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                # Title
                title_el = article_el.find(".//ArticleTitle")
                title = _extract_text(title_el) if title_el is not None else ""

                # Abstract
                abstract_parts = []
                for abs_text in article_el.findall(".//AbstractText"):
                    label = abs_text.get("Label", "")
                    text = _extract_text(abs_text)
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                # Authors
                authors = []
                for author in article_el.findall(".//Author"):
                    last = author.findtext("LastName", "")
                    fore = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {fore}".strip())

                # Journal
                journal = article_el.findtext(".//Journal/Title", "")
                pub_year = article_el.findtext(".//PubDate/Year", "")

                # DOI
                doi = ""
                for eid in article_el.findall(".//ArticleId"):
                    if eid.get("IdType") == "doi":
                        doi = eid.text or ""
                        break

                if abstract and len(abstract.strip()) > 50:
                    articles.append(PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract.strip(),
                        authors=authors[:5],  # top 5 authors
                        journal=journal,
                        pub_year=pub_year,
                        doi=doi,
                    ))

        except Exception as e:
            logger.error("PubMed XML batch %d/%d failed: %s", i + 1, len(batches), e)

    return articles


def _extract_text(el: ET.Element) -> str:
    """Extract all text from an XML element including inline children."""
    return "".join(el.itertext()).strip()


# ── PMC Full-Text Fetch ───────────────────────────────────────────


async def _fetch_pmc_full_text(
    client: httpx.AsyncClient,
    article: PubMedArticle,
    email: str,
    rate_limit_delay: float,
) -> bool:
    """Fetch full-text from PMC for a single article. Returns True if successful."""
    if not article.pmcid:
        return False

    try:
        await asyncio.sleep(rate_limit_delay)
        resp = await client.get(
            f"{PUBMED_BASE}/efetch.fcgi",
            params={
                "db": "pmc",
                "id": article.pmcid,
                "rettype": "xml",
                "retmode": "xml",
                "email": email,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        sections: dict[str, str] = {}
        full_parts: list[str] = []

        # Extract body sections
        for sec in root.findall(".//body//sec"):
            sec_title_el = sec.find("title")
            sec_title = _extract_text(sec_title_el) if sec_title_el is not None else "Untitled Section"

            paragraphs = []
            for p in sec.findall(".//p"):
                text = _extract_text(p)
                if text:
                    paragraphs.append(text)

            if paragraphs:
                section_text = " ".join(paragraphs)
                sections[sec_title] = section_text
                full_parts.append(f"## {sec_title}\n{section_text}")

        # If no structured sections, try to get all body text
        if not full_parts:
            for p in root.findall(".//body//p"):
                text = _extract_text(p)
                if text:
                    full_parts.append(text)

        if full_parts:
            article.full_text = "\n\n".join(full_parts)
            article.sections = sections
            article.full_text_source = "pmc"
            return True

    except Exception as e:
        logger.debug("PMC full-text fetch failed for %s: %s", article.pmcid, e)

    return False


# ── Europe PMC Fallback ───────────────────────────────────────────


async def _fetch_europepmc_full_text(
    client: httpx.AsyncClient,
    article: PubMedArticle,
    rate_limit_delay: float,
) -> bool:
    """Try fetching full-text from Europe PMC as fallback. Returns True if successful."""
    if not article.pmcid:
        return False

    try:
        await asyncio.sleep(rate_limit_delay)
        resp = await client.get(
            f"{EUROPE_PMC_BASE}/{article.pmcid}/fullTextXML",
            timeout=60.0,
        )
        if resp.status_code != 200:
            return False

        root = ET.fromstring(resp.text)
        sections: dict[str, str] = {}
        full_parts: list[str] = []

        for sec in root.findall(".//body//sec"):
            sec_title_el = sec.find("title")
            sec_title = _extract_text(sec_title_el) if sec_title_el is not None else "Untitled Section"

            paragraphs = []
            for p in sec.findall(".//p"):
                text = _extract_text(p)
                if text:
                    paragraphs.append(text)

            if paragraphs:
                section_text = " ".join(paragraphs)
                sections[sec_title] = section_text
                full_parts.append(f"## {sec_title}\n{section_text}")

        if not full_parts:
            for p in root.findall(".//body//p"):
                text = _extract_text(p)
                if text:
                    full_parts.append(text)

        if full_parts:
            article.full_text = "\n\n".join(full_parts)
            article.sections = sections
            article.full_text_source = "europepmc"
            return True

    except Exception as e:
        logger.debug("Europe PMC fallback failed for %s: %s", article.pmcid, e)

    return False


# ── Section-Based Chunking ────────────────────────────────────────

# Sections prioritised for medical Q&A generation
_PRIORITY_SECTIONS = [
    "results", "discussion", "methods", "introduction",
    "conclusion", "background", "findings",
]


def chunk_article_by_sections(
    article: PubMedArticle,
    max_chunk_words: int = 500,
    overlap_words: int = 50,
) -> list[tuple[str, str]]:
    """Chunk an article intelligently by section boundaries.

    Returns list of (chunk_text, section_name) tuples.
    Prioritises Results/Discussion sections for medical QA quality.
    """
    if not article.has_full_text or not article.sections:
        # Fallback: return entire content as a single "abstract" chunk
        content = article.best_content
        if content:
            return [(content, "abstract")]
        return []

    # Sort sections: priority sections first, then others
    def section_priority(name: str) -> int:
        lower = name.lower()
        for i, prio in enumerate(_PRIORITY_SECTIONS):
            if prio in lower:
                return i
        return len(_PRIORITY_SECTIONS)

    sorted_sections = sorted(article.sections.items(), key=lambda x: section_priority(x[0]))

    chunks: list[tuple[str, str]] = []
    for sec_name, sec_text in sorted_sections:
        words = sec_text.split()
        if len(words) <= max_chunk_words:
            chunks.append((sec_text, sec_name))
        else:
            # Split long sections into overlapping chunks
            for start in range(0, len(words), max_chunk_words - overlap_words):
                chunk_words = words[start:start + max_chunk_words]
                if len(chunk_words) < 50:
                    continue  # skip tiny trailing chunks
                chunk_text = " ".join(chunk_words)
                chunks.append((chunk_text, sec_name))

    return chunks


# ── Main Public API ───────────────────────────────────────────────


async def fetch_pubmed_abstracts(
    medical_terms: list[str],
    email: str = "user@example.com",
    retmax: int = 1000,
    batch_size: int = 100,
    rate_limit_delay: float = 0.34,
    fetch_full_text: bool = True,
    max_full_text: int = 50,
) -> PubMedResult:
    """Fetch articles from PubMed with optional full-text retrieval.

    Parameters
    ----------
    medical_terms : list of keywords to search
    email : required by NCBI for tracking
    retmax : maximum number of articles to fetch
    batch_size : IDs per efetch request
    rate_limit_delay : seconds between requests (NCBI limit: 3/sec)
    fetch_full_text : whether to attempt full-text retrieval from PMC
    max_full_text : max number of articles to fetch full-text for (rate limit protection)
    """
    result = PubMedResult(abstracts=[])

    term_queries = []
    for term in medical_terms:
        t = term.strip()
        if t:
            escaped = t.replace('"', '\\"')
            term_queries.append(f'"{escaped}"[Title/Abstract]')

    if not term_queries:
        logger.info("No medical terms provided for PubMed search.")
        return result

    query = "(" + " OR ".join(term_queries) + ")"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Step 1: Search for IDs (prefer free full text when available)
        try:
            # First search for free full text articles
            ft_query = f"{query} AND free full text[Filter]"
            search_resp = await client.get(
                f"{PUBMED_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": ft_query,
                    "retmax": retmax,
                    "retmode": "json",
                    "email": email,
                },
            )
            search_resp.raise_for_status()
            ft_data = search_resp.json()
            ft_ids = ft_data.get("esearchresult", {}).get("idlist", [])
            logger.info("PubMed free full-text search: %d IDs", len(ft_ids))

            # Then search without filter to get remaining
            await asyncio.sleep(rate_limit_delay)
            search_resp = await client.get(
                f"{PUBMED_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": retmax,
                    "retmode": "json",
                    "email": email,
                },
            )
            search_resp.raise_for_status()
            all_data = search_resp.json()
            all_ids = all_data.get("esearchresult", {}).get("idlist", [])

            # Merge: free full-text IDs first, then others
            ft_id_set = set(ft_ids)
            other_ids = [pid for pid in all_ids if pid not in ft_id_set]
            id_list = ft_ids + other_ids
            id_list = id_list[:retmax]

            result.total_ids_found = len(id_list)
            logger.info("PubMed combined: %d total IDs (%d free full-text)",
                        len(id_list), len(ft_ids))

        except Exception as e:
            logger.error("PubMed search error: %s", e)
            result.errors.append(f"Search failed: {e}")
            return result

        if not id_list:
            return result

        # Step 2: Fetch structured abstracts with metadata (XML)
        articles = await _fetch_abstracts_xml(
            client, id_list, email, rate_limit_delay, batch_size,
        )
        logger.info("Fetched %d articles with abstracts", len(articles))

        # Step 3: Convert PMIDs to PMCIDs for full-text retrieval
        if fetch_full_text and articles:
            pmid_list = [a.pmid for a in articles if a.pmid]
            pmid_to_pmcid = await _convert_pmids_to_pmcids(
                client, pmid_list, email, rate_limit_delay,
            )
            for article in articles:
                if article.pmid in pmid_to_pmcid:
                    article.pmcid = pmid_to_pmcid[article.pmid]

            # Step 4: Fetch full-text for articles with PMCIDs
            articles_with_pmc = [a for a in articles if a.pmcid][:max_full_text]

            # Fetch full-text in parallel (3 concurrent) instead of one-by-one
            _ft_sem = asyncio.Semaphore(3)

            async def _fetch_one_fulltext(art):
                async with _ft_sem:
                    if await _fetch_pmc_full_text(client, art, email, rate_limit_delay):
                        return True
                    if await _fetch_europepmc_full_text(client, art, rate_limit_delay):
                        return True
                    return False

            ft_results = await asyncio.gather(
                *[_fetch_one_fulltext(a) for a in articles_with_pmc],
                return_exceptions=True,
            )
            ft_success = sum(1 for r in ft_results if r is True)

            logger.info("Full-text: %d/%d articles retrieved successfully",
                        ft_success, len(articles_with_pmc))

        # Build results
        for article in articles:
            if article.has_full_text:
                result.full_text_count += 1
            else:
                result.abstract_only_count += 1
            result.abstracts.append(article.best_content)
            result.articles.append(article)

        result.total_fetched = len(articles)

    logger.info(
        "PubMed fetch complete: %d articles (%d full-text, %d abstract-only) from %d IDs",
        result.total_fetched, result.full_text_count, result.abstract_only_count,
        result.total_ids_found,
    )
    return result
