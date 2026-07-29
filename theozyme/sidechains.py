"""Build real catalytic side chains onto the scaffold so the output is a mutated
structure, not a set of target markers."""
import numpy as np
from .geometry import kabsch, rotate_about, dihedral

# (chi index -> rotation bond, atoms that move)
CHI_DEF = {
 'LYS': [(('CA','CB'), ['CG','CD','CE','NZ']),
         (('CB','CG'), ['CD','CE','NZ']),
         (('CG','CD'), ['CE','NZ']),
         (('CD','CE'), ['NZ'])],
 'TYR': [(('CA','CB'), ['CG','CD1','CD2','CE1','CE2','CZ','OH']),
         (('CB','CG'), ['CD1','CD2','CE1','CE2','CZ','OH'])],
 'HIS': [(('CA','CB'), ['CG','ND1','CD2','CE1','NE2']),
         (('CB','CG'), ['ND1','CD2','CE1','NE2'])],
 'GLU': [(('CA','CB'), ['CG','CD','OE1','OE2']),
         (('CB','CG'), ['CD','OE1','OE2']),
         (('CG','CD'), ['OE1','OE2'])],
 'ASP': [(('CA','CB'), ['CG','OD1','OD2']), (('CB','CG'), ['OD1','OD2'])],
 'ASN': [(('CA','CB'), ['CG','OD1','ND2']), (('CB','CG'), ['OD1','ND2'])],
 'GLN': [(('CA','CB'), ['CG','CD','OE1','NE2']),
         (('CB','CG'), ['CD','OE1','NE2']), (('CG','CD'), ['OE1','NE2'])],
 'SER': [(('CA','CB'), ['OG'])],
 'THR': [(('CA','CB'), ['OG1','CG2'])],
 'CYS': [(('CA','CB'), ['SG'])],
}
TIP = {'LYS':'NZ','TYR':'OH','HIS':'NE2','GLU':'OE1','ASP':'OD1',
       'ASN':'ND2','GLN':'NE2','SER':'OG','THR':'OG1','CYS':'SG'}

def get_template(st, resn, chain=None):
    """Pull a native residue of this type out of the scaffold to use as geometry template."""
    chain = chain or st.chain[0]
    best=None
    for ch,r in st.protein_res:
        if st.resname(r,ch)!=resn: continue
        idx=st.residues[(ch,r)]
        names=[st.name[i] for i in idx]
        need=['N','CA','C','CB']+[a for _,mv in CHI_DEF[resn] for a in mv]
        if not all(a in names for a in need): continue
        d={st.name[i]: st.xyz[i] for i in idx}
        b=float(np.mean([st.b[i] for i in idx]))
        if best is None or b<best[0]: best=(b,d)
    return best[1] if best else None

def build_sidechain(st, resi, resn, target_atom=None, target_xyz=None, chain=None,
                    template=None, coarse=15.0, refine_rounds=3):
    """Graft a `resn` side chain onto residue `resi`, optionally driving its tip atom
    onto target_xyz by chi optimisation. Returns {atom_name: coords} or None."""
    chain = chain or st.chain[0]
    tpl = template or get_template(st, resn, chain)
    if tpl is None: return None
    host = {nm: st.atom(resi, nm, chain) for nm in ('N','CA','C','CB')}
    if host['N'] is None or host['CA'] is None or host['C'] is None: return None
    if host['CB'] is None:                      # GLY host: build an ideal CB
        n,ca,c = host['N'],host['CA'],host['C']
        b1=n-ca; b2=c-ca
        bis=-(b1/np.linalg.norm(b1)+b2/np.linalg.norm(b2))
        # cross(b1,b2), not cross(b2,b1): the flipped form puts CB on the mirror
        # side of the N-CA-C plane, i.e. builds a D-amino acid (improper N-C-CA-CB
        # -121 deg instead of +121, 2.38 A from the correct position).
        perp=np.cross(b1,b2)
        v=bis/np.linalg.norm(bis)*np.cos(np.radians(50.0)) + perp/np.linalg.norm(perp)*np.sin(np.radians(50.0))
        host['CB']=ca+1.521*v/np.linalg.norm(v)
    R,t = kabsch(np.array([tpl['N'],tpl['CA'],tpl['CB']]),
                 np.array([host['N'],host['CA'],host['CB']]))
    sc = {nm: R@xyz+t for nm,xyz in tpl.items()}
    sc['N'],sc['CA'],sc['C'],sc['CB'] = host['N'],host['CA'],host['C'],host['CB']
    if target_atom is None or target_xyz is None:
        return sc
    chis = CHI_DEF[resn]
    def apply(state, angles):
        s = {k:v.copy() for k,v in state.items()}
        for (b0,b1),moving in zip([c[0] for c in chis], [c[1] for c in chis]):
            pass
        return s
    def build(angles):
        s = {k:v.copy() for k,v in sc.items()}
        for ((b0,b1),moving),th in zip(chis, angles):
            o,ax = s[b0], s[b1]-s[b0]
            pts=np.array([s[m] for m in moving])
            pts=rotate_about(pts,o,ax,np.radians(th))
            for m,p in zip(moving,pts): s[m]=p
        return s
    n=len(chis); best=(1e9,[0.0]*n)
    grid=np.arange(-180,180,coarse)
    # greedy per-chi optimisation, then joint refinement
    ang=[0.0]*n
    for _ in range(refine_rounds):
        for k in range(n):
            vals=[]
            for g in grid:
                a=list(ang); a[k]=g
                vals.append((np.linalg.norm(build(a)[target_atom]-target_xyz), g))
            e,g=min(vals); ang[k]=g
        step=coarse/4
        for k in range(n):
            vals=[]
            for g in np.arange(ang[k]-coarse, ang[k]+coarse+1e-9, step):
                a=list(ang); a[k]=g
                vals.append((np.linalg.norm(build(a)[target_atom]-target_xyz), g))
            e,g=min(vals); ang[k]=g
        coarse=max(coarse/3, 0.5)
    s=build(ang)
    err=float(np.linalg.norm(s[target_atom]-target_xyz))
    s['_chi']=[float(x) for x in ang]; s['_err']=err
    return s

def mutate(st, resi, resn, coords, chain=None):
    """Replace residue `resi` in-place with `resn` and the supplied side-chain coords."""
    chain = chain or st.chain[0]
    idx = st.residues[(chain,int(resi))]
    keep_bb = [i for i in idx if st.name[i] in ('N','CA','C','O')]
    order = ['N','CA','C','O','CB'] + [a for _,mv in CHI_DEF[resn] for a in mv]
    seen=set(); order=[a for a in order if not (a in seen or seen.add(a))]
    new=[]
    for nm in order:
        if nm in ('N','CA','C','O'):
            src=[i for i in keep_bb if st.name[i]==nm]
            if not src: continue
            new.append((nm, st.xyz[src[0]], st.elem[src[0]], st.b[src[0]]))
        elif nm in coords:
            new.append((nm, coords[nm], nm[0], 20.0))
    return dict(resi=int(resi), resn=resn, atoms=new)

def apply_mutations(st, muts, chain=None):
    """Return a new Structure with the listed residues replaced."""
    from .structure import Structure
    chain = chain or st.chain[0]
    drop=set()
    for m in muts: drop.update(st.residues[(chain,m['resi'])])
    out=Structure()
    rec=[];name=[];resn=[];ch=[];resi=[];occ=[];b=[];elem=[];xyz=[]
    bymut={m['resi']:m for m in muts}
    done=set()
    for i in range(len(st.xyz)):
        r=int(st.resi[i])
        if i in drop:
            if r in done: continue
            done.add(r); m=bymut[r]
            for nm,q,el,bb in m['atoms']:
                rec.append('ATOM'); name.append(nm); resn.append(m['resn']); ch.append(chain)
                resi.append(r); occ.append(1.0); b.append(float(bb)); elem.append(el); xyz.append(q)
            continue
        rec.append(st.rec[i]); name.append(st.name[i]); resn.append(st.resn[i])
        ch.append(st.chain[i]); resi.append(r); occ.append(1.0); b.append(st.b[i])
        elem.append(st.elem[i]); xyz.append(st.xyz[i])
    out.rec=np.array(rec); out.name=np.array(name); out.resn=np.array(resn)
    out.chain=np.array(ch); out.resi=np.array(resi); out.occ=np.array(occ)
    out.b=np.array(b); out.elem=np.array(elem); out.xyz=np.array(xyz)
    out._index(); return out




RING_EQUIV = [('CD1','CD2'), ('CE1','CE2')]  # default; see RING_EQUIV_BY_RES

# Dunbrack rotamer wells, chi1 then chi2. Only residues whose side chain has a
# meaningful rotamer preference are listed; anything absent is left unrestricted.
ROTAMER_WELLS = {
 'TYR': ([-177.0, -65.0, 62.0], [-85.0, -30.0, 80.0]),
 'PHE': ([-177.0, -65.0, 62.0], [-85.0, -30.0, 80.0]),
 'HIS': ([-177.0, -65.0, 62.0], [-165.0, -80.0, 60.0, 80.0]),
 'TRP': ([-177.0, -65.0, 62.0], [-105.0, -90.0, -4.0, 95.0]),
 'ASP': ([-177.0, -70.0, 62.0], [-15.0, 0.0, 30.0]),
 'ASN': ([-177.0, -65.0, 62.0], [-80.0, -20.0, 30.0, 120.0]),
 'LEU': ([-177.0, -65.0, 62.0], [65.0, 175.0]),
 'MET': ([-177.0, -65.0, 62.0], [-177.0, -65.0, 65.0, 180.0]),
 'GLU': ([-177.0, -67.0, 62.0], [-177.0, -65.0, 65.0, 180.0]),
 'GLN': ([-177.0, -67.0, 62.0], [-177.0, -65.0, 65.0, 180.0]),
 'SER': ([-177.0, -65.0, 62.0],),
 'THR': ([-177.0, -60.0, 62.0],),
 'CYS': ([-177.0, -65.0, 62.0],),
 'LYS': ([-177.0, -67.0, 62.0], [-177.0, -68.0, 68.0, 180.0]),
 'ARG': ([-177.0, -67.0, 62.0], [-177.0, -67.0, 65.0, 180.0]),
 'VAL': ([-60.0, 175.0, 63.0],),
 'ILE': ([-177.0, -65.0, 62.0], [-60.0, 170.0, 100.0]),
}

def off_rotamer(value, wells):
    """Smallest angular distance from `value` to any well, in degrees."""
    return min(abs((value - w + 180.0) % 360.0 - 180.0) for w in wells)

def fit_ring(st, resi, resn, target, chain=None, template=None, step=2.0,
             wells=True, well_tol=(30.0, 40.0),
             tip_target=None, tip_weight=1.0):
    """Build a CHEMICALLY VALID side chain on the scaffold backbone, fitted to a
    target geometry.

    Unlike grafting the QM atoms verbatim, this keeps CA-CB at its proper length --
    the mismatch is reported as an RMSD.

    Returns the side chain dict with _ring_rmsd, _chi, _off_rotamer and, when a tip
    target is given, _tip_dist and _tip_dev.
    """
    chain = chain or st.chain[0]
    tpl = template or get_template(st, resn, chain)
    base = build_sidechain(st, resi, resn, chain=chain, template=tpl)
    if base is None: return None
    chis = CHI_DEF[resn]
    if not chis: return base
    names = [a for a in target if a in base]
    eq = RING_EQUIV_BY_RES.get(resn, [])
    W = ROTAMER_WELLS.get(resn) if wells else None

    def build(angles):
        s = {k: v.copy() for k, v in base.items()}
        for ((b0, b1), moving), th in zip(chis, angles):
            o, ax = s[b0], s[b1] - s[b0]
            P = rotate_about(np.array([s[m] for m in moving]), o, ax, np.radians(th))
            for m, p in zip(moving, P): s[m] = p
        return s

    def ring_rmsd(s):
        e1 = np.sqrt(np.mean([np.sum((s[n] - target[n])**2) for n in names]))
        if not eq: return e1
        sw = dict(target)
        for a, b in eq:
            if a in sw and b in sw: sw[a], sw[b] = sw[b], sw[a]
        e2 = np.sqrt(np.mean([np.sum((s[n] - sw[n])**2) for n in names]))
        return min(e1, e2)

    def penalty(angles):
        if W is None: return 0.0
        p = 0.0
        for k, (a, wk) in enumerate(zip(angles, W)):
            tol = well_tol[k] if k < len(well_tol) else well_tol[-1]
            d = off_rotamer(a, wk)
            if d > tol: return np.inf
            p += d
        return p

    def cost(s, angles):
        pen = penalty(angles)
        if not np.isfinite(pen): return np.inf, None
        if tip_target is None:
            return ring_rmsd(s) + 0.002 * pen, None
        atom, xyz, want = tip_target
        d = float(np.linalg.norm(s[atom] - np.asarray(xyz)))
        return tip_weight * abs(d - want) + 0.05 * ring_rmsd(s) + 0.002 * pen, d

    grid = np.arange(-180.0, 180.0, step)
    best = (np.inf, None, None, None)
    if len(chis) == 1:
        for c in grid:
            s = build([c]); v, d = cost(s, [c])
            if v < best[0]: best = (v, s, [float(c)], d)
    else:
        for c1 in grid:
            for c2 in grid:
                s = build([c1, c2]); v, d = cost(s, [c1, c2])
                if v < best[0]: best = (v, s, [float(c1), float(c2)], d)
        if len(chis) > 2:                      # refine trailing chis greedily
            ang = best[2] + [0.0] * (len(chis) - 2)
            for _ in range(3):
                for k in range(2, len(chis)):
                    vals = []
                    for g in grid:
                        a = list(ang); a[k] = float(g)
                        s = build(a); v, d = cost(s, a)
                        vals.append((v, g, s, d))
                    v, g, s, d = min(vals, key=lambda z: z[0])
                    ang[k] = float(g); best = (v, s, list(ang), d)
    if best[1] is None:
        if W is not None:                      # nothing on-rotamer: retry unrestricted
            return fit_ring(st, resi, resn, target, chain, template, step,
                            wells=False, tip_target=tip_target, tip_weight=tip_weight)
        return None
    _, s, ang, tipd = best
    s['_ring_rmsd'] = float(ring_rmsd(s))
    s['_chi'] = [float(x) for x in ang]
    s['_off_rotamer'] = ([round(off_rotamer(a, w), 1) for a, w in zip(ang, W)]
                         if W is not None else None)
    if tipd is not None:
        s['_tip_dist'] = round(float(tipd), 2)
        s['_tip_dev'] = round(abs(float(tipd) - tip_target[2]), 2)
    return s


# ---------------------------------------------------------------- extended library
CHI_DEF.update({
 'ARG': [(('CA','CB'), ['CG','CD','NE','CZ','NH1','NH2']),
         (('CB','CG'), ['CD','NE','CZ','NH1','NH2']),
         (('CG','CD'), ['NE','CZ','NH1','NH2']),
         (('CD','NE'), ['CZ','NH1','NH2'])],
 'TRP': [(('CA','CB'), ['CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2']),
         (('CB','CG'), ['CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'])],
 'PHE': [(('CA','CB'), ['CG','CD1','CD2','CE1','CE2','CZ']),
         (('CB','CG'), ['CD1','CD2','CE1','CE2','CZ'])],
 'MET': [(('CA','CB'), ['CG','SD','CE']), (('CB','CG'), ['SD','CE']),
         (('CG','SD'), ['CE'])],
 'LEU': [(('CA','CB'), ['CG','CD1','CD2']), (('CB','CG'), ['CD1','CD2'])],
 'ILE': [(('CA','CB'), ['CG1','CG2','CD1']), (('CB','CG1'), ['CD1'])],
 'VAL': [(('CA','CB'), ['CG1','CG2'])],
 'ALA': [],
})
TIP.update({'ARG':'NH1','TRP':'NE1','PHE':'CZ','MET':'SD','LEU':'CG',
            'ILE':'CD1','VAL':'CB','ALA':'CB'})
RING_EQUIV_BY_RES = {'TYR':[('CD1','CD2'),('CE1','CE2')],
                     'PHE':[('CD1','CD2'),('CE1','CE2')],
                     'ASP':[('OD1','OD2')],'GLU':[('OE1','OE2')],
                     'ARG':[('NH1','NH2')],'LEU':[('CD1','CD2')],'VAL':[('CG1','CG2')]}


def solve_segment_shift(st, resi, resn, tip_atom, tip_target, chain=None,
                        template=None, wells=True, well_tol=(25.0, 35.0),
                        chi_step=3.0, partner=None, target_dist=None):
    """
    Smallest RIGID TRANSLATION of a residue that lets an ON-ROTAMER side chain
    place `tip_atom` exactly on `tip_target`.
    """
    import numpy as np
    chain = chain or st.chain[0]
    tpl = template or get_template(st, resn, chain)
    base = build_sidechain(st, resi, resn, chain=chain, template=tpl)
    if base is None or tip_atom not in base: return None
    chis = CHI_DEF[resn]
    if not chis: return None
    W = ROTAMER_WELLS.get(resn) if wells else None

    def build(angles):
        s = {k: v.copy() for k, v in base.items()}
        for ((b0, b1), moving), th in zip(chis, angles):
            o, ax = s[b0], s[b1] - s[b0]
            P = rotate_about(np.array([s[m] for m in moving]), o, ax, np.radians(th))
            for m, p in zip(moving, P): s[m] = p
        return s

    grid = np.arange(-180.0, 180.0, chi_step)
    combos = ([[c] for c in grid] if len(chis) == 1
              else [[a, b] for a in grid for b in grid])
    best = None
    for ang in combos:
        if W is not None:
            bad = False
            for k, (a, wk) in enumerate(zip(ang, W)):
                tol = well_tol[k] if k < len(well_tol) else well_tol[-1]
                if off_rotamer(a, wk) > tol: bad = True; break
            if bad: continue
        tip = build(ang)[tip_atom]
        if partner is not None and target_dist is not None:
            # DISTANCE target: the catalytic constraint is a separation, not a point.
            # Requiring the tip to sit on an exact coordinate is stricter than the
            # chemistry and inflates the shift (2.83 A vs 0.0 for RA95 TYR51).
            # For a tip at p, the smallest translation giving |p+d-partner| = target
            # is along the line through partner and p.
            pv = tip - np.asarray(partner, float)
            cur = float(np.linalg.norm(pv))
            mag = abs(cur - target_dist)
            shift = (pv / max(cur, 1e-9)) * (target_dist - cur)
        else:
            shift = np.asarray(tip_target, float) - tip
            mag = float(np.linalg.norm(shift))
        if best is None or mag < best[0]:
            best = (mag, shift, [float(x) for x in ang])
    if best is None: return None
    mag, shift, ang = best
    return dict(shift=shift, magnitude=round(mag, 3), chi=[round(x, 1) for x in ang],
                off_rotamer=([round(off_rotamer(a, w), 1) for a, w in zip(ang, W)]
                             if W is not None else None))
