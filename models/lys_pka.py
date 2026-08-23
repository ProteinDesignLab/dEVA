import os
import io
import math
import logging
import tempfile
import warnings
import contextlib
from typing import Dict

from common.utils import ensure_dir
from core.interfaces import BaseModel
from core.registry import register_model
from evolve.individual import Individual

logger = logging.getLogger("evolution")
logger.setLevel(logging.DEBUG)

IONIZABLE = {"ASP", "GLU", "HIS", "ARG", "LYS"}
AMIDE = {"ASN": ("OD1", "ND2"), "GLN": ("OE1", "NE2")}
MODEL_PKA_LYS = 10.50


def _get(cfg, key, default=None):
    """Config access that works for dicts and attribute-style config objects.

    Deliberately broad: some config wrappers raise KeyError from __getattr__
    rather than returning a default, so both lookups are guarded.
    """
    if cfg is None:
        return default
    val = None
    try:
        val = cfg[key]
    except Exception:                                             # noqa: BLE001
        try:
            val = getattr(cfg, key)
        except Exception:                                         # noqa: BLE001
            val = None
    return default if val is None else val


@register_model("lys_pka")
class LysPkaModel(BaseModel):
    """Catalytic-lysine pKa objective for retro-aldolase design.

    score = pka_gate * (env + amide + burial)      higher is better

    Calibrated on the RA95 evolution series (4A29 -> 4A2S -> 4A2R -> 5AN7):
    every functional variant has an apo Lys pKa of 7.0-8.5 and NO ionizable
    sidechain within 4.5 A of NZ.  5AN7 scores ~1.39 under this function.

    The catalytic Lys is resolved from the fixed-residue list rather than
    hard-coded, so one block serves all five theozyme placements.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------ setup

    @staticmethod
    def _silence_propka(level=logging.CRITICAL):
        """PROPKA logs its banner, citations and per-residue warnings through the
        logging module, which bypasses redirect_stdout entirely -- on a 60x30 run
        that is ~1800 copies of the banner in the evolution log.

        Two layers, because the propka.* submodule loggers do NOT exist yet when
        setup() runs -- they are created lazily on first use, so anything that
        only walks logging.root.manager.loggerDict silences nothing:
          1. detach the PARENT 'propka' logger (propagate=False stops children)
          2. filter propka records at every root handler, which survives even if
             something later re-enables propagation
        Set models.lys_pka.propka_log_level to INFO/DEBUG to see it again.
        """
        for name in list(logging.root.manager.loggerDict):
            if name == "propka" or name.startswith("propka."):
                lg = logging.getLogger(name)
                lg.setLevel(level)
                lg.propagate = False
                lg.handlers = [logging.NullHandler()]
        parent = logging.getLogger("propka")
        parent.setLevel(level)
        parent.propagate = False
        parent.handlers = [logging.NullHandler()]

        if level > logging.DEBUG:
            class _NoPropka(logging.Filter):
                def filter(self, record):
                    return not record.name.startswith("propka")
            for handler in logging.getLogger().handlers:
                if not any(isinstance(f, _NoPropka) for f in handler.filters):
                    handler.addFilter(_NoPropka())

    def setup(self, config: Dict, device: str = "cpu") -> None:
        try:
            import propka.run  # noqa: F401
        except ImportError:
            raise ImportError("lys_pka requires propka -- pip install propka")

        lvl = _get(_get(_get(config, "models"), "lys_pka"), "propka_log_level", "CRITICAL")
        self._propka_level = getattr(logging, str(lvl).upper(), logging.CRITICAL)
        self._silence_propka(self._propka_level)

        self.config = config
        self.device = device

        self.pdb = self.config.input.pdb
        self.pdb_name = os.path.split(self.pdb)[-1].split(".")[0]
        self.model_config = self.config.models.lys_pka

        mc = self.model_config
        self.chain = _get(mc, "chain", "A")
        self.ignore_ligand = bool(_get(mc, "ignore_ligand", True))
        self.keep_het = tuple(_get(mc, "keep_het", []) or [])

        self.pka_lo = float(_get(mc, "pka_lo", 7.0))
        self.pka_hi = float(_get(mc, "pka_hi", 8.5))
        self.pka_floor = float(_get(mc, "pka_floor", 6.0))
        self.pka_ceil = float(_get(mc, "pka_ceil", 10.0))
        self.gate_mode = _get(mc, "gate_mode", "multiply")

        self.ion_cut = float(_get(mc, "ion_cut", 4.5))
        self.ion_weight = float(_get(mc, "ion_weight", 1.0))
        self.carboxylate_factor = float(_get(mc, "carboxylate_factor", 1.5))
        self.ion_residues = set(_get(mc, "ion_residues", sorted(IONIZABLE)))

        self.amide_lo = float(_get(mc, "amide_lo", 2.5))
        self.amide_hi = float(_get(mc, "amide_hi", 3.6))
        self.amide_opt = float(_get(mc, "amide_opt", 2.9))
        self.amide_weight = float(_get(mc, "amide_weight", 0.6))

        self.burial_weight = float(_get(mc, "burial_weight", 0.4))
        self.burial_ref_shift = float(_get(mc, "burial_ref_shift", 2.93))

        # 0.0 = no grid snapping; the raw float is reported.  Rounding to 0.05
        # collapsed designs differing by real amounts into ties (a sum of
        # -0.018 printed as 0.00), which flattens NSGA-II selection.
        self.round_to = float(_get(mc, "round_to", 0.0))
        self.fail_value = float(_get(mc, "fail_value", -5.0))
        # ONE fitness key only.  Every key passed to add_fitness() becomes an
        # NSGA-II objective, so emitting the component terms as fitness would
        # silently turn a 4-objective run into a 10-objective one and destroy
        # selection pressure.  Components go to a CSV instead.
        self.write_components = bool(_get(mc, "write_components", True))
        self.verbose = bool(_get(mc, "verbose", False))

        self.lys_resi = self._resolve_lys()

        if bool(_get(mc, "require_lys", True)):
            self._assert_is_lys(self.pdb, self.lys_resi)

        outputs = self.config.general.outputs
        self.output_pka = os.path.join(outputs, "lys_pka")
        ensure_dir(self.output_pka)
        self.components_csv = os.path.join(self.output_pka, "lys_pka_components.csv")
        if self.write_components and not os.path.exists(self.components_csv):
            with open(self.components_csv, "w") as fh:
                fh.write("gen,index,structure,score,apo_pka,in_window,gate,env,"
                         "amide,burial,n_ionizable,ionizable,amide_contact\n")

        logger.info(
            "lys_pka [quiet-propka build]: catalytic Lys%d, apo=%s, window %.1f-%.1f",
            self.lys_resi, self.ignore_ligand,
            self.pka_lo, self.pka_hi)

    def _resolve_lys(self):
        """lys_resi -> fixed_residues[lys_index] -> theozyme_residues[lys_index]."""
        mc = self.model_config
        li = int(_get(mc, "lys_index", 0))

        lys = _get(mc, "lys_resi")
        if lys is not None:
            return int(lys)

        def num(tok):
            digits = "".join(c for c in tok if c.isdigit())
            return int(digits) if digits else None

        src = _get(mc, "fixed_from", "ligandmpnn")
        block = _get(self.config.models, src)
        fixed = _get(block, "fixed_residues", "")
        toks = [t.strip() for t in str(fixed).split() if t.strip()]
        if len(toks) > li:
            lys = num(toks[li])
            if lys is not None:
                logger.info("lys_pka: resolved Lys%d from %s.fixed_residues %s",
                            lys, src, toks)
                return lys

        theo = _get(_get(self.config.models, "relax")
                    or _get(self.config.models, "protpardelle_relax"),
                    "theozyme_residues", [])
        theo = list(theo)
        if len(theo) > li:
            logger.info("lys_pka: resolved Lys%d from theozyme_residues",
                        int(theo[li]))
            return int(theo[li])

        raise ValueError(
            "lys_pka: could not resolve the catalytic Lys. Set lys_resi "
            "explicitly, or make sure fixed_residues / theozyme_residues "
            "list the Lys first.")

    def _assert_is_lys(self, path, resi):
        for resn, num, name, _ in self._atoms(path):
            if num == resi and name == "CA":
                if resn != "LYS":
                    raise ValueError(
                        f"lys_pka: residue {resi} in {path} is {resn}, not LYS. "
                        f"Check the order of fixed_residues -- the catalytic "
                        f"lysine must come first, or set lys_resi explicitly.")
                return
        raise ValueError(f"lys_pka: residue {resi} not found in {path}")

    # ------------------------------------------------------------ helpers

    def _atoms(self, path, het=False):
        out = []
        with open(path, errors="replace") as fh:
            for line in fh:
                if not (line.startswith("ATOM") or (het and line.startswith("HETATM"))):
                    continue
                if self.chain and line[21] not in (self.chain, " "):
                    continue
                if line[16] not in " A1":
                    continue
                try:
                    num = int(line[22:26])
                except ValueError:
                    continue
                out.append((line[17:20].strip(), num, line[12:16].strip(),
                            (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
        return out

    def _strip_het(self, path):
        fd, tmp = tempfile.mkstemp(suffix=".pdb", prefix="lyspka_")
        with os.fdopen(fd, "w") as out, open(path, errors="replace") as fh:
            for line in fh:
                if line.startswith("HETATM") and line[17:20].strip() not in self.keep_het:
                    continue
                out.write(line)
        return tmp

    def _propka(self, path):
        import propka.run as pk
        # propka.run imports submodules lazily, so loggers can appear after setup
        self._silence_propka(self._propka_level)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol = pk.single(path, optargs=["-q"], write_pka=False)
            groups = mol.conformations["AVR"].groups
        for g in groups:
            try:
                if g.residue_type == "LYS" and int(g.atom.res_num) == self.lys_resi:
                    return float(g.pka_value)
            except (TypeError, ValueError):
                continue
        return None

    # ------------------------------------------------------------ terms

    def _pka_gate(self, pka):
        if self.pka_lo <= pka <= self.pka_hi:
            return 1.0
        if pka < self.pka_lo:
            return max(0.0, (pka - self.pka_floor) / (self.pka_lo - self.pka_floor))
        return max(0.0, (self.pka_ceil - pka) / (self.pka_ceil - self.pka_hi))

    def _env(self, atoms, nz):
        hits = {}
        for resn, num, name, xyz in atoms:
            if num == self.lys_resi or name in ("N", "O") or not name or name[0] not in "NO":
                continue
            if resn not in self.ion_residues:
                continue
            d = math.dist(xyz, nz)
            if d >= self.ion_cut:
                continue
            key = f"{resn}{num}"
            if key not in hits or d < hits[key]:
                hits[key] = d
        penalty = 0.0
        for key, d in hits.items():
            w = self.carboxylate_factor if key[:3] in ("ASP", "GLU") else 1.0
            penalty += w * (self.ion_cut - d) / (self.ion_cut - 2.5)
        return -self.ion_weight * penalty, hits

    def _amide(self, atoms, nz):
        best, who = 0.0, ""
        for resn, num, name, xyz in atoms:
            if num == self.lys_resi or resn not in AMIDE or name not in AMIDE[resn]:
                continue
            d = math.dist(xyz, nz)
            if self.amide_lo <= d <= self.amide_hi:
                s = 1.0 - abs(d - self.amide_opt) / (self.amide_hi - self.amide_lo)
                if s > best:
                    best, who = s, f"{resn}{num}.{name}@{d:.2f}"
        return self.amide_weight * best, who

    def _burial(self, pka):
        shift = MODEL_PKA_LYS - pka
        if shift <= 0:
            return 0.0
        return self.burial_weight * min(1.0, shift / self.burial_ref_shift)

    def _log_components(self, gen, index, path, score, pka, gate, env, amide,
                        burial, ion, who):
        """Diagnostics go to CSV, never to add_fitness -- see write_components."""
        try:
            row = [str(gen), str(index), os.path.basename(path),
                   repr(float(score)), repr(float(pka)),
                   str(int(self.pka_lo <= pka <= self.pka_hi)), repr(float(gate)),
                   repr(float(env)), repr(float(amide)),
                   repr(float(burial)),
                   str(len(ion)),
                   ";".join(f"{k}@{v}" for k, v in sorted(ion.items(),
                                                          key=lambda x: x[1])),
                   who or ""]
            with open(self.components_csv, "a") as fh:
                fh.write(",".join(row) + "\n")
        except Exception as exc:                                  # noqa: BLE001
            logger.debug("lys_pka: could not write components row: %s", exc)

    # ------------------------------------------------------------ score

    def score(self, individual: Individual):
        index = individual.get_index()
        gen = individual.get_gen()
        curr_pdb = individual.get_name()

        fitness = {"lys_pka_score": self.fail_value}

        tmp = None
        try:
            atoms = self._atoms(curr_pdb)
            nz = next((a[3] for a in atoms
                       if a[1] == self.lys_resi and a[2] == "NZ"), None)
            if nz is None:
                logger.warning("lys_pka: no NZ at residue %d in %s (gen%s ind%s)",
                               self.lys_resi, curr_pdb, gen, index)
                individual.add_fitness(fitness)
                return

            target = curr_pdb
            if self.ignore_ligand:
                tmp = self._strip_het(curr_pdb)
                target = tmp

            pka = self._propka(target)
            if pka is None:
                logger.warning("lys_pka: propka found no LYS%d in %s",
                               self.lys_resi, curr_pdb)
                individual.add_fitness(fitness)
                return

            gate = self._pka_gate(pka)
            env, ion = self._env(atoms, nz)
            amide, who = self._amide(atoms, nz)
            burial = self._burial(pka)

            geom = env + amide + burial
            score = gate * geom if self.gate_mode == "multiply" else gate + geom

            fitness["lys_pka_score"] = float(score)

            if self.write_components:
                self._log_components(
                    gen, index, curr_pdb, score, pka, gate, env, amide,
                    burial, ion, who)

            if self.verbose:
                logger.info(
                    "lys_pka gen%s ind%s: pKa %r gate %r env %r (%s) "
                    "amide %r (%s) burial %r -> %r",
                    gen, index, pka, gate, env,
                    ",".join(f"{k}@{v}" for k, v in sorted(ion.items(), key=lambda x: x[1]))
                    or "clean",
                    amide, who or "-",
                    burial, score)

        except Exception as exc:                                  # noqa: BLE001
            logger.warning("lys_pka failed on gen%s ind%s (%s): %s",
                           gen, index, curr_pdb, exc)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

        individual.add_fitness(fitness)