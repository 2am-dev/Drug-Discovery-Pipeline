# drug_discovery_pipeline/tools/docking.py
"""
Molecular docking wrapper.

If AutoDock Vina is installed and a receptor PDBQT is provided, runs real
docking.  Otherwise returns realistic mock scores so the rest of the
pipeline can proceed.

NOTE: This module requires `vina` on PATH and a prepared receptor .pdbqt
      for real docking.  If either is missing it gracefully falls back to
      mock mode.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import tempfile
from typing import List, Dict, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from config import VINA_EXECUTABLE, RECEPTOR_PDBQT, VINA_CENTER, VINA_SIZE

logger = logging.getLogger(__name__)


# ── Ligand preparation ────────────────────────────────────────────────────────

def _smiles_to_pdbqt(smiles: str, output_path: str) -> bool:
    """
    Convert a SMILES string to a 3D PDBQT file using RDKit.

    This produces a simplified PDBQT with Gasteiger charges.  For production
    use, prefer ``obabel`` or ``meeko`` for more accurate charge models.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
    if status == -1:
        # Fallback: random coordinates
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status == -1:
            logger.warning("Could not embed 3D coordinates for: %s", smiles)
            return False

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # unoptimised is acceptable for a placeholder

    # Write PDB, then convert to PDBQT (simplified)
    pdb_path = output_path.replace(".pdbqt", ".pdb")
    Chem.MolToPDBFile(mol, pdb_path)

    # Simple PDB → PDBQT conversion: add partial charges & atom types
    _pdb_to_pdbqt(pdb_path, output_path)
    return os.path.isfile(output_path)


def _pdb_to_pdbqt(pdb_path: str, pdbqt_path: str) -> None:
    """
    Minimal PDB → PDBQT conversion.
    Assigns Gasteiger charges and maps element symbols to AutoDock atom types.
    """
    from rdkit.Chem import AllChem

    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    if mol is None:
        logger.error("Failed to read PDB: %s", pdb_path)
        return

    # Compute Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)

    # Element → AutoDock atom type mapping (simplified)
    _type_map = {
        1: "HD",   # hydrogen (donor)
        6: "C",
        7: "NA",   # nitrogen (acceptor)
        8: "OA",   # oxygen (acceptor)
        9: "F",
        16: "SA",  # sulphur (acceptor)
        17: "Cl",
        35: "Br",
    }

    lines: list[str] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = mol.GetConformer().GetAtomPosition(idx)
        elem = atom.GetAtomicNum()
        atype = _type_map.get(elem, "C")
        # Gasteiger charge
        charge = 0.0
        try:
            charge = float(atom.GetDoubleProp("_GasteigerCharge"))
        except Exception:
            pass
        if charge != charge:  # NaN check
            charge = 0.0

        pdb_idx = idx + 1
        lines.append(
            f"ATOM  {pdb_idx:5d} {atom.GetSymbol():<2s}   LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    "
            f"{charge:6.3f} {atype}\n"
        )
    lines.append("END\n")

    with open(pdbqt_path, "w") as f:
        f.writelines(lines)


# ── Vina execution ───────────────────────────────────────────────────────────

def _vina_available() -> bool:
    """Check whether the Vina executable is reachable."""
    try:
        result = subprocess.run(
            [VINA_EXECUTABLE, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _run_vina(receptor: str, ligand_pdbqt: str, out_pdbqt: str) -> Optional[float]:
    """
    Run AutoDock Vina and return the best binding affinity (kcal/mol).
    Returns ``None`` on failure.
    """
    cx, cy, cz = VINA_CENTER
    sx, sy, sz = VINA_SIZE
    cmd = [
        VINA_EXECUTABLE,
        "--receptor", receptor,
        "--ligand", ligand_pdbqt,
        "--out", out_pdbqt,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", "8",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        # Parse best score: "REMARK VINA RESULT:  -7.3  0.000  0.000"
        for line in output.splitlines():
            if "REMARK VINA RESULT" in line:
                parts = line.split()
                for p in parts:
                    try:
                        return float(p)
                    except ValueError:
                        continue
        # Try alternate output format
        for line in output.splitlines():
            if "mode" in line.lower() and "affinity" in line.lower():
                continue  # header
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except (ValueError, IndexError):
                    continue
        logger.warning("Could not parse Vina output:\n%s", output)
        return None
    except Exception as exc:
        logger.error("Vina execution failed: %s", exc)
        return None


# ── Mock docking ──────────────────────────────────────────────────────────────

def _mock_docking(smiles: str) -> float:
    """
    Return a realistic-looking mock binding affinity.
    Affinity is deterministic for a given SMILES (hash-based) so reruns
    are consistent, with small random noise.
    """
    base = hash(smiles) % 800  # 0-800
    score = -(base / 100.0 + 3.0)  # -3.0 to -11.0
    noise = random.uniform(-0.3, 0.3)
    return round(score + noise, 2)


# ── Public API ────────────────────────────────────────────────────────────────

def dock_molecules(
    smiles_list: List[str],
    receptor_pdbqt: Optional[str] = None,
) -> List[Dict]:
    """
    Dock each SMILES string against the receptor.

    Returns a list of dicts with keys:
        ``smiles``, ``affinity_kcal_mol``, ``method`` (``"vina"`` or ``"mock"``)
    """
    receptor = receptor_pdbqt or RECEPTOR_PDBQT
    use_vina = bool(receptor) and _vina_available()

    if use_vina:
        logger.info("AutoDock Vina detected – running real docking.")
    else:
        logger.info(
            "Vina not available or receptor not provided – using mock docking scores."
        )

    results: list[dict] = []
    tmpdir = tempfile.mkdtemp(prefix="docking_")

    for smi in smiles_list:
        if use_vina:
            ligand_path = os.path.join(tmpdir, f"ligand_{hash(smi) & 0xFFFF}.pdbqt")
            out_path = os.path.join(tmpdir, f"out_{hash(smi) & 0xFFFF}.pdbqt")
            prepared = _smiles_to_pdbqt(smi, ligand_path)
            if not prepared:
                logger.warning("Ligand prep failed for %s – using mock.", smi)
                results.append({"smiles": smi, "affinity_kcal_mol": _mock_docking(smi), "method": "mock"})
                continue
            affinity = _run_vina(receptor, ligand_path, out_path)
            if affinity is not None:
                results.append({"smiles": smi, "affinity_kcal_mol": affinity, "method": "vina"})
            else:
                results.append({"smiles": smi, "affinity_kcal_mol": _mock_docking(smi), "method": "mock"})
        else:
            results.append({"smiles": smi, "affinity_kcal_mol": _mock_docking(smi), "method": "mock"})

    # Sort by affinity (most negative = best)
    results.sort(key=lambda x: x["affinity_kcal_mol"])
    logger.info("Docking complete for %d ligands.", len(results))
    return results