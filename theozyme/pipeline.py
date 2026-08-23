"""End-to-end light relaxation for a theozyme-bearing design."""
import os

import numpy as np

from .structure import Structure
from .rigid import RigidTheozyme
from .accommodate import propagate_sidechains_cb
from .accommodate import (accommodate, pick_mobile, all_atom_clearance,
                          count_sidechain_clashes)


class TheozymeRelaxPipeline:
    """Reusable relaxer. Build once, call `run()` per individual."""

    def __init__(self, reference_pdb, theozyme_resis, ligand_resi=None, chain='A',
                 backend='protpardelle', relaxer=None, mode='fixed',
                 clearance=3.40, shell=10.0, mobile_resis=None,
                 reseat_warn=1.5, accept_tol=0.05, max_backbone_disp=None, bond_tol=0.02,
                 n_protpardelle_attempts=1, select_by_sc_clashes=True,
                 accommodate_kw=None, verbose=False):
        self.chain = chain
        self.backend = backend
        self.relaxer = relaxer
        self.mode = mode
        self.clearance = float(clearance)
        self.shell = float(shell)
        self.reseat_warn = float(reseat_warn)
        self.accept_tol = float(accept_tol)
        self.bond_tol = float(bond_tol)
        self.max_backbone_disp = (None if max_backbone_disp in (None, 0)
                                  else float(max_backbone_disp))
        self.n_protpardelle_attempts = max(1, int(n_protpardelle_attempts))
        # Among gate-passing PD draws: prefer fewest post-propagate SC–ligand
        # clashes, then SC–protein, then lowest bb_rmsd ("least aggressive"
        # rewind — not ML likelihood).
        self.select_by_sc_clashes = bool(select_by_sc_clashes)
        self.accommodate_kw = dict(accommodate_kw or {})
        self.verbose = verbose

        if not os.path.isfile(reference_pdb):
            raise FileNotFoundError(f'reference pdb not found: {reference_pdb}')
        self.ref_st = Structure(reference_pdb)
        self.rigid = RigidTheozyme(self.ref_st, theozyme_resis,
                                   ligand_resi=ligand_resi, chain=chain)
        self.theozyme_resis = self.rigid.resis
        self.fixed_mobile = None if mobile_resis is None else [int(r) for r in mobile_resis]
        if self.verbose:
            print(self.rigid.summary())

        if backend == 'protpardelle' and relaxer is not None:
            problems = relaxer.preflight()
            if problems and self.verbose:
                print('[pipeline] protpardelle unavailable: ' + '; '.join(problems))

    def _check_compatible(self, st):
        """The individual must still carry the theozyme residues we froze.

        Returns a list of problem strings. Categories:
          - residue missing / wrong amino acid (fatal for this individual)
          - missing side-chain atoms (often packer incompleteness; may be grafted)
          - ligand absent (often keep_ligand_in_packed; may be grafted)
        """
        bad = []
        n_ref = len(self.ref_st.protein_res)
        n_got = len(st.protein_res)
        if n_got != n_ref:
            bad.append(f'protein length {n_got} != reference {n_ref}')
        for r, rr in self.rigid.residues.items():
            key = (self.chain, r)
            if key not in st.residues:
                bad.append(f'residue {self.chain}{r} ({rr.resn}) missing')
                continue
            idx = st.residues[key]
            got = str(st.resn[idx[0]])
            if got != rr.resn:
                bad.append(f'residue {self.chain}{r} is {got}, reference has {rr.resn} '
                           f'-- pin it in ligandmpnn.fixed_residues '
                           f'(e.g. "{self.chain}{r}")')
                continue
            names = {str(st.name[i]) for i in idx}
            missing = [n for n in rr.atoms if n not in names]
            if missing:
                bad.append(f'residue {self.chain}{r} ({rr.resn}) missing atoms {missing}')
        if self.rigid.lig_key is not None and self.rigid.lig_key not in st.residues:
            bad.append(f'ligand {self.rigid.lig_key} absent -- check '
                       f'ligandmpnn.keep_ligand_in_packed')
        return bad

    def _format_compat_error(self, pdb_in, problems):
        """Multi-line ValueError body so the problems list is obvious in logs."""
        lines = [
            f'theozyme compatibility check failed for {pdb_in}',
            f'  ({len(problems)} problem(s); relax cannot freeze the catalytic geometry)',
        ]
        for p in problems:
            lines.append(f'  - {p}')
        return '\n'.join(lines)

    def _reseat(self, st, xyz):
        """Put the reference theozyme back, and measure how far it had drifted."""
        before = xyz.copy()
        out = self.rigid.restore(st, xyz)
        idx = self.rigid.atom_indices(st)
        shift = float(np.linalg.norm(out[idx] - before[idx], axis=1).max()) if len(idx) else 0.0
        return out, shift

    def _gates(self, acc, base_d):
        """Theozyme / clearance / disp / bond gates shared by PD and geometric."""
        ca = acc.get('clearance_after')
        ok = acc['theozyme']['ok']
        better = (base_d is None or ca is None
                  or ca['min_dist'] >= base_d - self.accept_tol)
        light = (self.max_backbone_disp is None
                 or acc['max_disp_mobile'] <= self.max_backbone_disp)
        bb, ba = acc.get('bonds_before'), acc.get('bonds_after')
        sane = (bb is None or ba is None
                or ba['worst_dev'] <= bb['worst_dev'] + self.bond_tol)
        return ok, better, light, sane, ca, bb, ba

    def _gate_fail_reason(self, ok, better, light, sane, acc, bb, ba):
        if not ok:
            return 'theozyme check failed'
        if not better:
            return 'clearance not improved'
        if not light:
            return (f'backbone moved {acc["max_disp_mobile"]:.2f} A '
                    f'> max_backbone_disp {self.max_backbone_disp}')
        return (f'backbone bonds degraded '
                f'{bb["worst_dev"]:.3f} -> {ba["worst_dev"]:.3f} A')

    def _finish_pp(self, st, xyz, mobile, proposed, pp_info):
        """Propagate → reseat → SC count → accommodate after one PD backbone."""
        work = _with(st, xyz)
        proposed = propagate_sidechains_cb(work, work.xyz, proposed, self.chain)
        proposed, _ = self._reseat(st, proposed)
        sc_clashes = count_sidechain_clashes(
            work, proposed, rigid=self.rigid, lig_clearance=self.clearance,
            chain=self.chain)
        work = _with(st, proposed)
        kw = dict(mode=self.mode, clearance=self.clearance, shell=self.shell,
                  verbose=self.verbose)
        kw.update(self.accommodate_kw)
        new, acc = accommodate(work, self.rigid, mobile_resis=mobile, **kw)
        return new, pp_info, acc, sc_clashes

    def _attempt(self, st, xyz, mobile, use_pp, seed):
        """One relaxation attempt (single PD draw or geometric-only).

        Returns (xyz, protpardelle_info, accommodate_report, sc_clashes_after_propagate).
        ``sc_clashes_after_propagate`` is scored right after PD + propagate (before
        geometric accommodate / rotamer relief) so ranking reflects backbone-only
        clash relief.
        """
        work = _with(st, xyz)
        if use_pp:
            problems = self.relaxer.preflight()
            if problems:
                raise RuntimeError('protpardelle unavailable: ' + '; '.join(problems))
            proposed, pp_info = self.relaxer.relax_structure(
                work, self.theozyme_resis, chain=self.chain, seed=seed)
            return self._finish_pp(st, xyz, mobile, proposed, pp_info)
        kw = dict(mode=self.mode, clearance=self.clearance, shell=self.shell,
                  verbose=self.verbose)
        kw.update(self.accommodate_kw)
        new, acc = accommodate(work, self.rigid, mobile_resis=mobile, **kw)
        return new, None, acc, None

    def _consider(self, plan, k, attempt_seed, new, pp_info, acc, sc_clashes,
                  base_d, report, viable, chosen):
        """Apply gates; update viable/chosen/fallbacks. Returns updated chosen."""
        ok, better, light, sane, ca, bb, ba = self._gates(acc, base_d)
        if ok and better and light and sane:
            if self.select_by_sc_clashes and plan == 'protpardelle':
                n_lig = (sc_clashes or {}).get('n_lig', 10 ** 9)
                n_pp = (sc_clashes or {}).get('n_protein', 10 ** 9)
                bb_rmsd = float((pp_info or {}).get('bb_rmsd', 99.0))
                viable.append((n_lig, n_pp, bb_rmsd, k, plan, new, pp_info, acc,
                               sc_clashes))
                if self.verbose:
                    print(f'[pipeline] PD attempt {k + 1} viable: '
                          f'sc_lig={n_lig} sc_pp={n_pp} bb_rmsd={bb_rmsd}',
                          flush=True)
                return chosen
            return (plan, new, pp_info, acc, sc_clashes, k + 1)
        report['fallbacks'].append(dict(
            plan=plan, attempt=k + 1, seed=attempt_seed,
            theozyme_ok=bool(ok),
            clearance=None if ca is None else ca['min_dist'],
            baseline=base_d,
            max_disp=acc['max_disp_mobile'],
            bond_dev=None if ba is None else ba['worst_dev'],
            sc_clashes=sc_clashes,
            bb_rmsd=None if pp_info is None else pp_info.get('bb_rmsd'),
            reason=self._gate_fail_reason(ok, better, light, sane, acc, bb, ba)))
        return chosen

    def run(self, pdb_in, pdb_out=None, seed=None):
        st = Structure(pdb_in)
        # Packer sometimes omits tip atoms / ligand; graft from the frozen
        # reference before treating that as a hard failure.
        grafted = self.rigid.graft_missing(st)
        if grafted and self.verbose:
            print('[pipeline] grafted missing theozyme atoms from reference: '
                  + ', '.join(grafted), flush=True)
        problems = self._check_compatible(st)
        if problems:
            raise ValueError(self._format_compat_error(pdb_in, problems))

        xyz, reseat = self._reseat(st, st.xyz.copy())
        report = dict(input=pdb_in, backend=self.backend,
                      theozyme_reseat_shift=round(reseat, 3),
                      reseat_warning=bool(reseat > self.reseat_warn),
                      grafted_from_reference=grafted,
                      fallbacks=[],
                      select_by_sc_clashes=self.select_by_sc_clashes)

        work = _with(st, xyz)
        if self.fixed_mobile is not None:
            mobile = [r for r in self.fixed_mobile if r not in set(self.theozyme_resis)]
        else:
            mobile, _, _ = pick_mobile(work, self.rigid, shell=self.shell,
                                       clearance=self.clearance, chain=self.chain)

        if self.backend not in ('protpardelle', 'geometric'):
            raise ValueError(f"backend must be 'protpardelle' or 'geometric', "
                             f"got {self.backend!r}")
        if self.backend == 'protpardelle' and self.relaxer is None:
            raise ValueError("backend='protpardelle' needs a relaxer")

        base = all_atom_clearance(work, xyz, self.rigid)
        base_d = base['min_dist'] if base else None

        # Default: first gate-passing attempt wins (PD draws, then geometric).
        # select_by_sc_clashes: evaluate all PD draws, rank by post-propagate SC
        # clashes then bb_rmsd, geometric only if none pass.
        # PD draws are batched in one model.sample(B=n_attempts) call.
        chosen = None
        viable = []
        if self.backend == 'protpardelle':
            problems = self.relaxer.preflight()
            if problems:
                raise RuntimeError('protpardelle unavailable: ' + '; '.join(problems))
            n_pp = self.n_protpardelle_attempts
            batch_seed = None if seed is None else int(seed)
            proposed, infos = self.relaxer.relax_structure(
                work, self.theozyme_resis, chain=self.chain, seed=batch_seed,
                n_samples=n_pp)
            if n_pp == 1:
                proposed, infos = [proposed], [infos]
            for k in range(n_pp):
                new, pp_info, acc, sc_clashes = self._finish_pp(
                    st, xyz, mobile, proposed[k], infos[k])
                chosen = self._consider(
                    'protpardelle', k, batch_seed, new, pp_info, acc, sc_clashes,
                    base_d, report, viable, chosen)
                if chosen is not None and not self.select_by_sc_clashes:
                    break
            if not self.select_by_sc_clashes and chosen is None:
                # Early-exit mode: geometric after all PD draws failed.
                attempt_seed = (None if seed is None
                                else int(seed) + 17 * n_pp)
                new, pp_info, acc, sc_clashes = self._attempt(
                    st, xyz, mobile, False, attempt_seed)
                chosen = self._consider(
                    'geometric', n_pp, attempt_seed, new, pp_info, acc,
                    sc_clashes, base_d, report, viable, chosen)
        else:
            attempt_seed = None if seed is None else int(seed)
            new, pp_info, acc, sc_clashes = self._attempt(
                st, xyz, mobile, False, attempt_seed)
            chosen = self._consider(
                'geometric', 0, attempt_seed, new, pp_info, acc, sc_clashes,
                base_d, report, viable, chosen)

        if chosen is None and viable:
            viable.sort()
            n_lig, n_pp_sc, bb_rmsd, k, plan, new, pp_info, acc, sc_clashes = viable[0]
            chosen = (plan, new, pp_info, acc, sc_clashes, k + 1)
            if self.verbose:
                print(f'[pipeline] selected PD attempt {k + 1}/{self.n_protpardelle_attempts} '
                      f'by sc_lig={n_lig}, sc_pp={n_pp_sc}, bb_rmsd={bb_rmsd} '
                      f'({len(viable)} viable)', flush=True)
            for alt in viable[1:]:
                an_lig, an_pp, abr, ak, aplan, _, _app, aacc, asc = alt
                report['fallbacks'].append(dict(
                    plan=aplan, attempt=ak + 1,
                    theozyme_ok=True,
                    clearance=(aacc.get('clearance_after') or {}).get('min_dist'),
                    baseline=base_d,
                    max_disp=aacc['max_disp_mobile'],
                    sc_clashes=asc,
                    bb_rmsd=abr,
                    reason=(f'out-ranked (sc_lig={an_lig}, sc_pp={an_pp}, '
                            f'bb_rmsd={abr})')))

        if chosen is None and self.select_by_sc_clashes and self.backend == 'protpardelle':
            # No PD draw passed gates — fall through to geometric once.
            attempt_seed = None if seed is None else int(seed) + 17 * self.n_protpardelle_attempts
            new, pp_info, acc, sc_clashes = self._attempt(
                st, xyz, mobile, False, attempt_seed)
            chosen = self._consider(
                'geometric', self.n_protpardelle_attempts, attempt_seed,
                new, pp_info, acc, sc_clashes, base_d, report, viable, chosen)

        if chosen is None:
            report.update(backend='none', accepted=False, ok=True,
                          accommodate=dict(clearance_before=base, clearance_after=base,
                                           theozyme=self.rigid.verify(work, xyz)))
            new = xyz
        else:
            plan, new, pp_info, acc, sc_clashes, n_att = chosen
            report.update(backend=plan, accepted=True, protpardelle=pp_info,
                          accommodate=acc, ok=bool(acc['theozyme']['ok']),
                          n_attempts=n_att,
                          sc_clashes_after_propagate=sc_clashes)

        if pdb_out:
            self.write(st, new, pdb_out, report)
            report['output'] = pdb_out
        return _with(st, new), report

    def write(self, st, xyz, path, report=None):
        """Write the relaxed structure, preserving the covalent LINK to the ligand."""
        out = _with(st, xyz)
        links = []
        if self.rigid.lig_key is not None and self.rigid.lig_owner is not None:
            r = self.rigid.lig_owner
            rr = self.rigid.residues[r]
            best = None
            for nm, p in rr.atoms.items():
                d = np.linalg.norm(self.rigid.lig_xyz - p, axis=1)
                k = int(np.argmin(d))
                if best is None or d[k] < best[0]:
                    best = (float(d[k]), nm, self.rigid.lig_names[k])
            if best and best[0] < 2.2:
                links.append(dict(
                    a=dict(name=best[1], resn=rr.resn, chain=self.chain, resi=r),
                    b=dict(name=best[2], resn=self.rigid.lig_resn,
                           chain=self.rigid.lig_key[0], resi=self.rigid.lig_key[1]),
                    dist=best[0]))
        remarks = ['relaxed by dEVA theozyme pipeline']
        if report:
            a = report.get('accommodate', {})
            t = a.get('theozyme', {})
            remarks += [
                f"backend={report.get('backend')} mode={a.get('mode')}",
                f"theozyme internal dev {t.get('max_internal_rmsd')} A, "
                f"locked-chi drift {t.get('max_locked_chi_drift')} deg, ok={t.get('ok')}"]
            ca = a.get('clearance_after')
            if ca:
                remarks.append(f"ligand clearance {ca['min_dist']} A at {ca['atom']}")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        out.write(path, remarks=remarks, links=links)
        return path


def _with(st, xyz):
    s = type(st).__new__(type(st))
    s.__dict__.update(st.__dict__)
    s.xyz = np.asarray(xyz, float)
    return s


def summarise(report):
    """One-line-per-stage digest for logs."""
    L = [f"[relax] {os.path.basename(report.get('input', '?'))}  "
         f"backend={report['backend']}  accepted={report.get('accepted')}  "
         f"ok={report.get('ok')}"]
    for f in report.get('fallbacks', []):
        L.append(f"  rejected {f['plan']}: {f['reason']} "
                 f"(clearance {f['clearance']} vs baseline {f['baseline']})")
    if report.get('reseat_warning'):
        L.append(f"  WARNING theozyme had drifted {report['theozyme_reseat_shift']} A "
                 f"before re-seating")
    pp = report.get('protpardelle')
    if pp:
        L.append(f"  protpardelle  noise={pp['noise_angstrom']} A  steps={pp['n_steps']}  "
                 f"bb_rmsd={pp['bb_rmsd']}  max={pp['bb_max_disp']}  "
                 f"superpose={pp['superpose_rmsd']}")
    sc = report.get('sc_clashes_after_propagate')
    if sc:
        L.append(f"  sc_clashes    lig={sc['n_lig']}  protein={sc['n_protein']}  "
                 f"total={sc['n_total']}  (post-propagate, pre-pack)")
    a = report.get('accommodate', {})
    if a and 'n_mobile' in a:
        L.append(f"  accommodate   mobile={a['n_mobile']}  rmsd(mobile)={a['rmsd_mobile']}  "
                 f"demand {a['substrate_demand_before']}->{a['substrate_demand_after']} A  "
                 f"tier {a['tier_before']}->{a['tier_after']}")
        cb, ca = a.get('clearance_before'), a.get('clearance_after')
        if cb and ca:
            L.append(f"  clearance     {cb['min_dist']} -> {ca['min_dist']} A "
                     f"(worst now {ca['atom']})")
        bb, ba = a.get('bonds_before'), a.get('bonds_after')
        if bb and ba:
            L.append(f"  bonds         worst backbone deviation "
                     f"{bb['worst_dev']} -> {ba['worst_dev']} A")
        t = a['theozyme']
        L.append(f"  theozyme      internal {t['max_internal_rmsd']:.1e} A  "
                 f"chi drift {t['max_locked_chi_drift']:.1e} deg")
    elif a:
        L.append('  no attempt was accepted; the input was returned unchanged')
    return '\n'.join(L)
