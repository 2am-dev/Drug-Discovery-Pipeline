# drug_discovery_pipeline/tools/target_lookup.py
# =============================================================================
# FILE: tools/target_lookup.py
# ROLE: Protein/target intelligence tool.
#       1. Search UniProt for protein entries matching a gene/disease query.
#       2. Retrieve protein metadata (sequence, function, disease associations).
#       3. Find PDB structural entries for the protein.
#       4. Download the PDB file to workdir for downstream docking preparation.
# =============================================================================

import logging
import os
from pathlib import Path
from typing import Optional

import requests

from config import PDB_BASE, PDB_SEARCH_BASE, UNIPROT_BASE, WORK_DIR

log = logging.getLogger("drug_discovery.target_lookup")

# ---------------------------------------------------------------------------
# UniProt REST helpers
# ---------------------------------------------------------------------------

def search_uniprot(query: str, max_results: int = 5) -> list[dict]:
    """
    Search UniProt for entries matching *query* (gene name or disease term).
    Returns a list of summary dicts for the top hits.
    """
    url = f"{UNIPROT_BASE}/search"
    params = {
        "query": query,
        "format": "json",
        "size": max_results,
        "fields": (
            "accession,gene_names,protein_name,organism_name,"
            "function_comment,disease_comment,length,sequence"
        ),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        proteins = []
        for r in results:
            accession = r.get("primaryAccession", "")
            gene_names = r.get("genes", [])
            gene_name = (
                gene_names[0].get("geneName", {}).get("value", "")
                if gene_names else ""
            )
            protein_name = (
                r.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "")
            )
            function_comments = [
                c.get("texts", [{}])[0].get("value", "")
                for c in r.get("comments", [])
                if c.get("commentType") == "FUNCTION"
            ]
            disease_comments = [
                c.get("disease", {}).get("diseaseId", "")
                for c in r.get("comments", [])
                if c.get("commentType") == "DISEASE"
            ]
            sequence_len = r.get("sequence", {}).get("length", 0)

            proteins.append(
                {
                    "uniprot_id": accession,
                    "gene_name": gene_name,
                    "protein_name": protein_name,
                    "organism": r.get("organism", {}).get("scientificName", ""),
                    "function": function_comments[0] if function_comments else "",
                    "diseases": disease_comments,
                    "sequence_length": sequence_len,
                }
            )
        log.info(f"UniProt search returned {len(proteins)} proteins.")
        return proteins
    except Exception as e:
        log.error(f"UniProt search failed: {e}")
        return []


def get_uniprot_details(uniprot_id: str) -> Optional[dict]:
    """
    Fetch full details for a specific UniProt accession.
    """
    url = f"{UNIPROT_BASE}/{uniprot_id}"
    params = {"format": "json"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"UniProt detail fetch failed for {uniprot_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# PDB structure helpers
# ---------------------------------------------------------------------------

def search_pdb_for_uniprot(uniprot_id: str) -> list[dict]:
    """
    Query RCSB PDB for structures linked to a given UniProt accession.
    Returns a list of PDB entry dicts: {pdb_id, title, resolution, method}.
    """
    query_body = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    try:
        resp = requests.post(PDB_SEARCH_BASE, json=query_body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result_set = data.get("result_set", [])
        entries = []
        for entry in result_set:
            pdb_id = entry.get("identifier", "")
            score = entry.get("score", 0)
            entries.append({"pdb_id": pdb_id, "score": score})
        log.info(f"PDB search found {len(entries)} structures for {uniprot_id}.")
        return entries
    except Exception as e:
        log.error(f"PDB search failed for UniProt {uniprot_id}: {e}")
        return []


def get_pdb_entry_info(pdb_id: str) -> Optional[dict]:
    """Fetch basic metadata for a PDB entry via RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return {
            "pdb_id": pdb_id.upper(),
            "title": data.get("struct", {}).get("title", ""),
            "resolution": data.get("rcsb_entry_info", {}).get(
                "resolution_combined", [None]
            )[0],
            "experimental_method": data.get("rcsb_entry_info", {}).get(
                "experimental_method", ""
            ),
            "deposition_date": data.get("rcsb_accession_info", {}).get(
                "deposit_date", ""
            ),
        }
    except Exception as e:
        log.warning(f"PDB entry info failed for {pdb_id}: {e}")
        return None


def download_pdb_file(pdb_id: str) -> Optional[Path]:
    """
    Download the PDB file for *pdb_id* into WORK_DIR.
    Returns the local path on success, None on failure.
    """
    pdb_id = pdb_id.upper()
    local_path = WORK_DIR / f"{pdb_id}.pdb"
    if local_path.exists():
        log.info(f"PDB file already cached: {local_path}")
        return local_path

    url = f"{PDB_BASE}/{pdb_id}.pdb"
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info(f"Downloaded PDB file: {local_path}")
        return local_path
    except Exception as e:
        log.error(f"PDB download failed for {pdb_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def lookup_target(query: str) -> dict:
    """
    Complete target lookup workflow:
    1. Search UniProt for proteins related to *query*.
    2. For the top hit, find PDB structures.
    3. Download the best-resolution PDB structure.

    Returns a comprehensive target info dict.
    """
    log.info(f"Looking up target for query: '{query}'")
    proteins = search_uniprot(query)
    if not proteins:
        log.warning("No UniProt entries found. Returning empty target info.")
        return {}

    # Filter to human proteins if possible
    human_proteins = [p for p in proteins if "Homo sapiens" in p.get("organism", "")]
    candidates = human_proteins or proteins

    # Take top candidate
    target = candidates[0].copy()
    target["all_candidates"] = candidates

    # Find PDB structures
    pdb_entries = search_pdb_for_uniprot(target["uniprot_id"])
    target["pdb_entries"] = pdb_entries

    # Get metadata for top PDB entries
    pdb_details = []
    for entry in pdb_entries[:5]:
        info = get_pdb_entry_info(entry["pdb_id"])
        if info:
            pdb_details.append(info)

    # Sort by resolution (lower is better)
    pdb_details.sort(
        key=lambda x: (x["resolution"] is None, x["resolution"] or 999)
    )
    target["pdb_details"] = pdb_details
    target["best_pdb"] = pdb_details[0]["pdb_id"] if pdb_details else ""
    target["pdb_id"] = target["best_pdb"]

    # Download PDB file
    if target["pdb_id"]:
        pdb_path = download_pdb_file(target["pdb_id"])
        target["pdb_local_path"] = str(pdb_path) if pdb_path else ""
    else:
        target["pdb_local_path"] = ""

    log.info(
        f"Target selected: {target.get('gene_name')} | "
        f"UniProt: {target.get('uniprot_id')} | "
        f"PDB: {target.get('pdb_id')}"
    )
    return target