# drug_discovery_pipeline/tools/target_lookup.py
"""
Target lookup tool – queries UniProt REST API for protein metadata and
finds available PDB structures.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional

import requests

from config import UNIPROT_BASE, PDB_DOWNLOAD_BASE

logger = logging.getLogger(__name__)


def lookup_target(target_name: str, limit: int = 5) -> List[Dict]:
    """
    Search UniProt for reviewed (Swiss-Prot) entries matching *target_name*.

    Returns a list of dicts with keys:
        ``accession``, ``gene``, ``protein_name``, ``organism``, ``pdb_ids``
    """
    params = {
        "query": f"{target_name} AND (reviewed:true)",
        "fields": "accession,gene_names,protein_name,organism_name,xref_pdb",
        "format": "json",
        "size": limit,
    }
    try:
        r = requests.get(f"{UNIPROT_BASE}/search", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.error("UniProt lookup failed for '%s': %s", target_name, exc)
        return []

    results: list[dict] = []
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession", "")
        genes = entry.get("genes", [])
        gene = genes[0].get("geneName", {}).get("value", "") if genes else ""
        protein = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
        organism = entry.get("organism", {}).get("scientificName", "")
        # PDB cross-references
        pdb_ids: list[str] = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "PDB":
                pdb_ids.append(xref.get("id", ""))
        results.append({
            "accession": acc,
            "gene": gene,
            "protein_name": protein,
            "organism": organism,
            "pdb_ids": pdb_ids,
        })
    logger.info("UniProt returned %d results for '%s'.", len(results), target_name)
    return results


def get_pdb_structure(uniprot_id: str) -> Optional[str]:
    """
    Given a UniProt accession, return the best available PDB ID
    (first listed), or ``None`` if no structure is available.
    """
    params = {
        "query": f"accession:{uniprot_id}",
        "fields": "xref_pdb",
        "format": "json",
        "size": 1,
    }
    try:
        r = requests.get(f"{UNIPROT_BASE}/search", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for entry in data.get("results", []):
            for xref in entry.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "PDB":
                    return xref["id"]
    except Exception as exc:
        logger.error("PDB lookup failed for UniProt %s: %s", uniprot_id, exc)
    return None