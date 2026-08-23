# models/pocket_shape.py
import os
import sys
import traceback
import numpy as np
from typing import Dict
from core.interfaces import BaseModel
from core.registry import register_model
from evolve.individual import Individual

BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def parse_pdb(pdb_path):
    """
    Parse a PDB into heavy-atom coordinate arrays.

    Returns:
        sidechain (n, 3), backbone (n, 3), hetatm (n, 3), hetatm_resnames (list)
    """
    sidechain, backbone, hetatm, resnames = [], [], [], []
    with open(pdb_path) as fh:
        for line in fh:
            record = line[:6]
            if record not in ("ATOM  ", "HETATM"):
                continue
            element = line[76:78].strip().upper() or line[12:16].strip()[0]
            if element == "H":
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if record == "HETATM":
                hetatm.append(xyz)
                resnames.append(line[17:20].strip())
            elif line[12:16].strip() in BACKBONE_ATOMS:
                backbone.append(xyz)
            else:
                sidechain.append(xyz)

    as_array = lambda a: np.asarray(a, dtype=np.float32).reshape(-1, 3)
    return as_array(sidechain), as_array(backbone), as_array(hetatm), resnames


@register_model("pocket_shape")
class PocketShape(BaseModel):
    """
    Substrate-aware geometric objective.

    Scores how well designed side chains enclose a fixed, mechanistically
    anchored ligand pose:

        fitness = window(occlusion) - w_clash * overlap - w_seal * excess_sealing

    Higher is better; the maximum of 1.0 is reached when occlusion equals
    target_occ. The window is deliberately not monotonic: maximising burial
    would seal the active site, which is incompatible with mechanisms
    requiring solvent access.

    target_occ is in RAW units -- the mean number of protein heavy atoms
    within r_occ of each ligand atom, typically 100-250 for a buried pose.
    Calibrate it against designs of known activity before running.
    """

    def __init__(self):
        pass

    def setup(self, config: Dict, device: str = "cpu") -> None:
        self.config = config
        self.device = device

        # Access this model's config section from YAML
        self.model_config = self.config.models.pocket_shape

        # Required parameter
        self.ligand_pdb = self.model_config.ligand_pdb

        # Optional parameters
        self.r_occ = float(self.model_config.get("r_occ", 8.0))
        self.softness = float(self.model_config.get("softness", 1.0))
        self.target_occ = float(self.model_config.get("target_occ", 150.0))
        self.r_clash = float(self.model_config.get("r_clash", 2.4))
        self.w_clash = float(self.model_config.get("w_clash", 1.0))
        self.w_seal = float(self.model_config.get("w_seal", 1.0))
        self.channel_max = float(self.model_config.get("channel_max", 0.25))
        self.channel_r = float(self.model_config.get("channel_r", 10.0))
        self.cone_cos = float(self.model_config.get("cone_cos", 0.7))
        self.verbose = bool(self.model_config.get("verbose", False))

        # Set up output directory
        outputs = self.config.general.outputs
        self.output_dir = os.path.join(outputs, "pocket_shape")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the fixed ligand pose (must share the frame of config.input.pdb)
        _, _, ligand, _ = parse_pdb(self.ligand_pdb)
        if len(ligand) == 0:
            raise ValueError(f"No HETATM heavy atoms found in {self.ligand_pdb}")
        self.ligand = ligand
        self.n_ligand = len(ligand)

        # Reference backbone occlusion. Kept for calibration and for the contact
        # sanity check below, but NOT reused per individual: with a relaxation stage
        # in the model list the backbone moves, and a cached value would describe the
        # input structure rather than the one being scored. See get_components().
        _, backbone, _, _ = parse_pdb(self.config.input.pdb)
        if len(backbone) == 0:
            raise ValueError(f"No backbone atoms parsed from {self.config.input.pdb}")
        self.backbone_occ = self._occlusion(backbone)

        # Guard against a ligand placed in a different coordinate frame
        separation = float(np.linalg.norm(self.ligand.mean(0) - backbone.mean(0)))
        if separation > 25.0:
            raise ValueError(
                f"Ligand centroid is {separation:.1f} A from the scaffold centroid. "
                f"Check that {self.ligand_pdb} and {self.config.input.pdb} share a frame.")
        if self.backbone_occ == 0.0:
            raise ValueError("Backbone occlusion is zero; ligand is not in contact.")

        # Solvent channel axis. For a buried site the centroid difference is
        # short and its direction is numerically meaningless, so prefer an
        # explicit axis (e.g. the metal -> nucleophile vector).
        self.ligand_centroid = self.ligand.mean(0)
        self.axis = None
        if self.w_seal > 0.0:
            axis = np.asarray(
                self.model_config.get("channel_axis", [0.0, 0.0, 0.0]), dtype=np.float32)
            if np.linalg.norm(axis) < 1e-6:
                axis = self.ligand_centroid - backbone.mean(0)
                if np.linalg.norm(axis) < 5.0:
                    raise ValueError(
                        "Ligand and scaffold centroids nearly coincide, so the implicit "
                        "channel axis is ill-conditioned. Set models.pocket_shape."
                        "channel_axis explicitly, or set w_seal: 0.0 to disable the "
                        "sealing term.")
            self.axis = axis / np.linalg.norm(axis)

    # ---------------------------------------------------------------- terms

    def _occlusion(self, coords):
        """Soft neighbour count summed over all (protein, ligand) atom pairs."""
        if len(coords) == 0:
            return 0.0
        d = np.sqrt(((coords[:, None, :] - self.ligand[None, :, :]) ** 2).sum(-1))
        # The whole quotient is summed. Attaching .sum() to the denominator
        # instead silently returns ~0 for every input.
        return float((1.0 / (1.0 + np.exp((d - self.r_occ) / self.softness))).sum())

    def _overlap(self, sidechain):
        """Total steric overlap in Angstroms, soft so marginal contacts cost little."""
        if len(sidechain) == 0:
            return 0.0
        d = np.sqrt(((sidechain[:, None, :] - self.ligand[None, :, :]) ** 2).sum(-1))
        return float(np.clip(self.r_clash - d, 0.0, None).sum())

    def _sealing(self, protein):
        """Fraction of protein atoms inside the outward cone. High means sealed."""
        v = protein - self.ligand_centroid
        d = np.linalg.norm(v, axis=1) + 1e-8
        in_cone = ((v @ self.axis) / d > self.cone_cos) & (d < self.channel_r)
        return float(in_cone.sum()) / max(len(protein), 1)

    def get_components(self, pdb_path):
        """Score a PDB and return the individual terms. Use to calibrate target_occ."""
        sidechain, backbone, _, _ = parse_pdb(pdb_path)
        protein = np.vstack([sidechain, backbone]) if len(sidechain) else backbone

        # Recomputed, not cached. Identical to self.backbone_occ when the backbone has
        # not moved, so existing target_occ calibrations remain valid; correct when it
        # has. Set pocket_shape.static_backbone: true to restore the old behaviour.
        if getattr(self, "static_backbone", False):
            backbone_occ = self.backbone_occ
        else:
            backbone_occ = self._occlusion(backbone)
        occlusion = (self._occlusion(sidechain) + backbone_occ) / self.n_ligand
        overlap = self._overlap(sidechain) / self.n_ligand
        sealing = self._sealing(protein) if self.axis is not None else 0.0

        fitness = 1.0 - abs(occlusion - self.target_occ) / max(self.target_occ, 1e-6)
        fitness -= self.w_clash * overlap
        fitness -= self.w_seal * max(0.0, sealing - self.channel_max)

        return {"occlusion": occlusion, "overlap": overlap,
                "sealing": sealing, "fitness": fitness}

    # ---------------------------------------------------------------- score

    def score(self, individual: Individual):
        pdb_path = individual.get_name()
        gen = individual.get_gen()
        index = individual.get_index()

        try:
            components = self.get_components(pdb_path)
            if components["occlusion"] == 0.0:
                raise ValueError("Zero occlusion; ligand is not in contact with protein.")
            score_value = components["fitness"]

            if self.verbose:
                print(f"[pocket_shape] gen {gen} idx {index}  "
                      f"occ={components['occlusion']:.1f}  "
                      f"overlap={components['overlap']:.3f}  "
                      f"seal={components['sealing']:.3f}  "
                      f"fitness={score_value:.3f}", flush=True)
        except Exception:
            # Log loudly. A silent fallback value hides real bugs for whole runs.
            print(f"[pocket_shape] scoring failed for {pdb_path}", file=sys.stderr)
            traceback.print_exc()
            score_value = -10.0

        individual.add_fitness({"pocket_shape": float(score_value)})