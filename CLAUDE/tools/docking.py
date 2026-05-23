# drug_discovery_pipeline/tools/docking.py
# =============================================================================
# FILE: tools/docking.py
# ROLE: Molecular docking wrapper tool.
#       Converts SMILES to 3D SDF/PDBQT using RDKit + Open Babel (if available).
#       Runs AutoDock Vina via subprocess.
#       Parses Vina output for binding energies.
#       Falls back to a realistic mock scorer if Vina/PDBQT prep fails.
# =============================================================================

import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from config import (
    DEFAULT_BOX_CENTER,
    DEFAULT_BOX_SIZE,
    RECEPTOR_PDBQT_PATH,
    VINA_ENERGY_RANGE,
    VINA_EXECUTABLE,
    VINA_EXHAUSTIVENESS,
    VINA_NUM_MODES,
    WORK_DIR,
)

log = logging.getLogger("drug_discovery.docking")


# ---------------------------------------------------------------------------
# Molecule 3D preparation
# ---------------------------------------------------------------------------

def smiles_to_3d_mol(smiles: str):
    """
    Convert a SMILES string to an RDKit Mol with 3D coordinates.
    Returns the Mol object or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.warning(f"Invalid SMILES: {smiles[:60]}")
        return None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Try ETKDG fallback
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result != 0:
        log.warning(f"3D embedding failed for SMILES: {smiles[:60]}")
        return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return mol


def mol_to_sdf(mol, path: Path) -> bool:
    """Write an RDKit Mol to SDF file. Returns True on success."""
    try:
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
        return True
    except Exception as e:
        log.error(f"SDF write failed: {e}")
        return False


def sdf_to_pdbqt(sdf_path: Path, pdbqt_path: Path) -> bool:
    """
    Convert SDF to PDBQT using Open Babel (obabel) via subprocess.
    Returns True on success.

    If obabel is not available, returns False (triggers mock docking).
    """
    if not shutil.which("obabel"):
        log.warning("obabel not found; cannot convert SDF → PDBQT.")
        return False
    try:
        cmd = [
            "obabel",
            str(sdf_path),
            "-O", str(pdbqt_path),
            "--gen3d",
            "-p", "7.4",  # pH for protonation
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
            return True
        log.warning(f"obabel conversion produced empty file: {result.stderr[:200]}")
        return False
    except Exception as e:
        log.error(f"obabel conversion failed: {e}")
        return False


def prepare_receptor_pdbqt(pdb_path: Path) -> Optional[Path]:
    """
    Prepare receptor PDBQT from a PDB file using AutoDock Tools (python-mgltools)
    or MGLTools prepare_receptor4.py.
    Falls back gracefully if not available.
    """
    pdbqt_path = pdb_path.with_suffix(".pdbqt")
    if pdbqt_path.exists():
        log.info(f"Receptor PDBQT already exists: {pdbqt_path}")
        return pdbqt_path

    # Try MGLTools prepare_receptor4.py
    prepare_script = shutil.which("prepare_receptor4.py")
    if prepare_script:
        try:
            cmd = [
                "python", prepare_script,
                "-r", str(pdb_path),
                "-o", str(pdbqt_path),
                "-A", "hydrogens",
                "-U", "nphs_lps_waters",
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if pdbqt_path.exists():
                return pdbqt_path
        except Exception as e:
            log.warning(f"prepare_receptor4.py failed: {e}")

    # Try obabel
    if shutil.which("obabel"):
        try:
            cmd = ["obabel", str(pdb_path), "-O", str(pdbqt_path), "-xr"]
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if pdbqt_path.exists():
                return pdbqt_path
        except Exception as e:
            log.warning(f"obabel receptor prep failed: {e}")

    log.warning("Could not prepare receptor PDBQT; will use mock docking.")
    return None


# ---------------------------------------------------------------------------
# Vina docking
# ---------------------------------------------------------------------------

def _parse_vina_output(output: str) -> Optional[float]:
    """
    Parse AutoDock Vina stdout and return the best binding energy (kcal/mol).
    Vina output table starts with a header row then mode rows:
        1      -8.5      0.000      0.000
    """
    for line in output.splitlines():
        line = line.strip()
        if re.match(r"^\s*1\s+", line):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return None


def run_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple = DEFAULT_BOX_CENTER,
    box_size: tuple = DEFAULT_BOX_SIZE,
) -> Optional[float]:
    """
    Run AutoDock Vina and return the best binding energy (kcal/mol).
    Returns None if Vina fails or is not installed.
    """
    if not shutil.which(VINA_EXECUTABLE):
        log.warning(f"Vina executable '{VINA_EXECUTABLE}' not found in PATH.")
        return None

    out_path = ligand_pdbqt.with_suffix(".out.pdbqt")
    cmd = [
        VINA_EXECUTABLE,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--exhaustiveness", str(VINA_EXHAUSTIVENESS),
        "--num_modes", str(VINA_NUM_MODES),
        "--energy_range", str(VINA_ENERGY_RANGE),
        "--out", str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        score = _parse_vina_output(result.stdout + result.stderr)
        if score is not None:
            log.info(f"Vina score for {ligand_pdbqt.name}: {score} kcal/mol")
        else:
            log.warning(f"Could not parse Vina score. Output:\n{result.stdout[:300]}")
        return score
    except subprocess.TimeoutExpired:
        log.error("Vina timed out.")
        return None
    except Exception as e:
        log.error(f"Vina execution failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Mock docking (placeholder)
# ---------------------------------------------------------------------------

def mock_docking_score(smiles: str, seed: int = 0) -> float:
    """
    *** PLACEHOLDER ***
    Return a realistic-looking mock docking score based on simple
    molecular descriptors as a proxy. Replace this with real Vina when
    a receptor PDBQT is available.

    Score range: -12 to -4 kcal/mol (typical Vina output range).
    """
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -4.0

    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rings = mol.GetRingInfo().NumRings()

    # Heuristic: heavier, more lipophilic, ring-rich → better (more negative) score
    base = -4.0
    base -= (mw / 500.0) * 2.0
    base -= (logp / 5.0) * 1.5
    base -= (rings / 3.0) * 1.0
    base -= (hba / 10.0) * 0.5

    # Add small reproducible noise
    rng = random.Random(hash(smiles) + seed)
    noise = rng.uniform(-0.5, 0.5)
    score = round(base + noise, 1)
    return max(-12.0, min(-4.0, score))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def dock_molecule(
    smiles: str,
    receptor_pdbqt_path: Optional[str] = None,
    center: tuple = DEFAULT_BOX_CENTER,
    box_size: tuple = DEFAULT_BOX_SIZE,
) -> dict:
    """
    Dock a single SMILES against the receptor.

    Workflow:
    1. Convert SMILES → 3D mol.
    2. Write SDF.
    3. Convert SDF → PDBQT (obabel).
    4. Run Vina (or fall back to mock).

    Returns a result dict: {smiles, score, mode, receptor}.
    """
    receptor_path = receptor_pdbqt_path or RECEPTOR_PDBQT_PATH
    mode = "vina"

    with tempfile.TemporaryDirectory(dir=WORK_DIR) as tmpdir:
        tmpdir = Path(tmpdir)

        mol = smiles_to_3d_mol(smiles)
        if mol is None:
            return {
                "smiles": smiles,
                "score": mock_docking_score(smiles),
                "mode": "mock_invalid_smiles",
                "receptor": receptor_path,
            }

        sdf_path = tmpdir / "ligand.sdf"
        pdbqt_path = tmpdir / "ligand.pdbqt"

        mol_to_sdf(mol, sdf_path)
        pdbqt_ok = sdf_to_pdbqt(sdf_path, pdbqt_path)

        score = None
        if pdbqt_ok and receptor_path and Path(receptor_path).exists():
            score = run_vina(
                receptor_pdbqt=Path(receptor_path),
                ligand_pdbqt=pdbqt_path,
                center=center,
                box_size=box_size,
            )

        if score is None:
            score = mock_docking_score(smiles)
            mode = "mock"
            log.info(f"Using mock docking score for {smiles[:40]}: {score}")

    return {
        "smiles": smiles,
        "score": score,
        "mode": mode,
        "receptor": receptor_path,
    }


def dock_multiple(
    smiles_list: list[str],
    receptor_pdbqt_path: Optional[str] = None,
    center: tuple = DEFAULT_BOX_CENTER,
    box_size: tuple = DEFAULT_BOX_SIZE,
) -> list[dict]:
    """
    Dock multiple SMILES and return sorted results (best score first).
    """
    results = []
    for i, smiles in enumerate(smiles_list):
        log.info(f"Docking molecule {i+1}/{len(smiles_list)}: {smiles[:50]}")
        result = dock_molecule(smiles, receptor_pdbqt_path, center, box_size)
        results.append(result)

    results.sort(key=lambda r: r["score"])  # most negative = best
    return results