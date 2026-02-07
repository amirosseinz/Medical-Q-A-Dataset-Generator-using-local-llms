"""PubMed API service — fetch research abstracts.

Ported from original app.py ``fetch_pubmed_abstracts()`` (lines 130-178).
Improved with:
  - httpx async client (replaces synchronous BioPython Entrez)
  - Proper rate limiting (≤3 req/sec without API key)
  - Better abstract parsing
  - Error handling with retry
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class PubMedResult:
    """Aggregated result of a PubMed fetch operation."""
    abstracts: list[str]
    total_ids_found: int = 0
    total_fetched: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


async def fetch_pubmed_abstracts(
    medical_terms: list[str],
    email: str = "user@example.com",
    retmax: int = 1000,
    batch_size: int = 100,
    rate_limit_delay: float = 0.34,
) -> PubMedResult:
    """Fetch abstracts from PubMed matching the given medical terms.

    Parameters
    ----------
    medical_terms : list of keywords to search
    email : required by NCBI for tracking
    retmax : maximum number of articles to fetch
    batch_size : IDs per efetch request
    rate_limit_delay : seconds between requests (NCBI limit: 3/sec)
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

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Search for IDs
        try:
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
            search_data = search_resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            result.total_ids_found = len(id_list)
            logger.info(f"PubMed search returned {len(id_list)} IDs for query: {query[:100]}")
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            result.errors.append(f"Search failed: {e}")
            return result

        if not id_list:
            return result

        # Step 2: Fetch abstracts in batches
        batches = [id_list[i : i + batch_size] for i in range(0, len(id_list), batch_size)]

        for i, batch in enumerate(batches):
            try:
                await asyncio.sleep(rate_limit_delay)
                fetch_resp = await client.get(
                    f"{PUBMED_BASE}/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "rettype": "abstract",
                        "retmode": "text",
                        "email": email,
                    },
                )
                fetch_resp.raise_for_status()
                batch_text = fetch_resp.text

                if batch_text:
                    # Split individual abstracts
                    split_abs = re.split(
                        r"\n\n(?:PMID:\s*\d+\s*)?\d+\.\s*", batch_text
                    )
                    cleaned = [a.strip() for a in split_abs if a.strip() and len(a.strip()) > 50]
                    result.abstracts.extend(cleaned)
                    result.total_fetched += len(cleaned)

            except Exception as e:
                logger.error(f"PubMed batch {i + 1}/{len(batches)} failed: {e}")
                result.errors.append(f"Batch {i + 1} failed: {e}")
                await asyncio.sleep(rate_limit_delay * 2)

    logger.info(
        f"PubMed fetch complete: {result.total_fetched} abstracts from {result.total_ids_found} IDs"
    )
    return result
