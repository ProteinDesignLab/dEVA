"""
Combinatorial position search: which scaffold positions can host which theozyme residues.
"""
import numpy as np, itertools, json
from collections import Counter, defaultdict
from scipy.spatial import cKDTree
from .geometry import place_atom, rotmat, angle

def _align(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b)
    v=np.cross(a,b); s=np.linalg.norm(v); c=float(np.dot(a,b))
    if s<1e-9: return np.eye(3) if c>0 else -np.eye(3)
    return rotmat(v, np.arctan2(s,c))

class CBIndex:
    """Scaffold CB positions with their CA directions, for satellite matching.

    `allowed` restricts which positions may host a satellite. Prefer
    prepare_deva --barrel-shell (ligand / strand C-mouth center) or an explicit
    --satellite-positions list; a mid-protein centroid is the wrong pocket.
    """
    def __init__(self, st, chain=None, exclude_resn=('GLY','PRO'), allowed=None):
        chain=chain or st.chain[0]
        allowed=set(int(x) for x in allowed) if allowed else None
        rs=[];ca=[];cb=[]
        for c,r in st.protein_res:
            if c!=chain: continue
            if allowed is not None and int(r) not in allowed: continue
            b=st.atom(r,'CB',c); a=st.atom(r,'CA',c)
            if b is None or a is None: continue
            rs.append(int(r)); ca.append(a); cb.append(b)
        self.resi=np.array(rs); self.CA=np.array(ca); self.CB=np.array(cb)
        self.tree=cKDTree(self.CB)
        self.wt={int(r):str(st.resname(r,chain)) for c,r in st.protein_res if c==chain}
    def match(self, target_cb, target_cg, max_dev=2.0, ang_tol=25.0, exclude=()):
        out=[]
        for i in self.tree.query_ball_point(target_cb, max_dev):
            r=int(self.resi[i])
            if r in exclude: continue
            a=angle(self.CA[i], self.CB[i], target_cg)
            if abs(a-114.1)>ang_tol: continue
            out.append(dict(resi=r, wt=self.wt[r],
                            cb_dev=float(np.linalg.norm(self.CB[i]-target_cb)),
                            cacbcg=float(a)))
        return sorted(out, key=lambda h:h['cb_dev'])

class Explorer:
    def __init__(self, spec, st, chain=None, mobile_resis=(),
                 clash_sub=3.20, clash_sc=2.70, max_cb_dev=2.0, ang_tol=25.0,
                 satellite_positions=None):
        self.spec=spec; self.st=st; self.chain=chain or st.chain[0]
        self.mobile=set(int(r) for r in mobile_resis)
        self.clash_sub=clash_sub; self.clash_sc=clash_sc
        self.max_cb_dev=max_cb_dev; self.ang_tol=ang_tol
        bb=st.backbone_idx(include_cb=True)
        keep=~np.isin(st.resi[bb], list(self.mobile))
        self.rigid=st.xyz[bb][keep]; self.rigid_res=st.resi[bb][keep]
        self.tree=cKDTree(self.rigid)
        self.satellite_positions=satellite_positions
        self.cbidx=CBIndex(st, self.chain, allowed=satellite_positions)
        a=spec.anchor
        self.iCB=a.atoms['CB']-1; self.iCG=a.atoms[a.cg]-1
        self.d_cbcg=float(np.linalg.norm(spec.X[self.iCG]-spec.X[self.iCB]))
        self.lig_idx=[spec.lig_atoms[k]-1 for k in spec.lig_atoms]
        self.lig_names=list(spec.lig_atoms)
        self.sat=[(s, s.atoms['CB']-1, s.atoms[s.cg]-1) for s in spec.satellites]
        self.anchor_sc=[v-1 for k,v in a.atoms.items() if k!='CB']

    def graft(self, N, CA, CB, chi1, chi2, ang_cacbcg=114.1):
        CGt=place_atom(N,CA,CB,self.d_cbcg,ang_cacbcg,chi1)
        X=self.spec.X - self.spec.X[self.iCB]
        X=(_align(self.spec.X[self.iCG]-self.spec.X[self.iCB], CGt-CB) @ X.T).T
        return (rotmat(CGt-CB, np.radians(chi2)) @ X.T).T + CB

    def run(self, anchor_positions, chi_step=3.0, require_all_satellites=True,
            progress=False):
        sols=[]
        chis=np.arange(-180,180,chi_step)
        for resi in anchor_positions:
            N,CA,CB=(self.st.atom(resi,x,self.chain) for x in ('N','CA','CB'))
            if CB is None: continue
            n_ok=0
            for c1 in chis:
                for c2 in chis:
                    X=self.graft(N,CA,CB,c1,c2)
                    sub=X[self.lig_idx]
                    d,i=self.tree.query(sub,k=8)
                    if np.where(self.rigid_res[i]==resi,1e3,d).min()<self.clash_sub: continue
                    sc=X[self.anchor_sc]
                    d,i=self.tree.query(sc,k=8)
                    if np.where(self.rigid_res[i]==resi,1e3,d).min()<self.clash_sc: continue
                    hosts=[]
                    ok=True
                    for s,icb,icg in self.sat:
                        h=self.cbidx.match(X[icb],X[icg],self.max_cb_dev,self.ang_tol,
                                           exclude=(resi,))
                        if not h: ok=False; break
                        hosts.append((s.resn,h))
                    if require_all_satellites and not ok: continue
                    for combo in itertools.product(*[h for _,h in hosts]) if hosts else [()]:
                        if len({c['resi'] for c in combo})<len(combo): continue
                        sols.append(dict(anchor=int(resi), anchor_wt=self.cbidx.wt.get(int(resi)),
                            anchor_resn=self.spec.anchor.resn, chi1=float(c1), chi2=float(c2),
                            satellites=[dict(resn=hosts[k][0], **combo[k]) for k in range(len(combo))],
                            X=X, sub=sub))
                        n_ok+=1
            if progress: print(f'  anchor {resi}: {n_ok} solutions')
        return sols

def summarise(sols, top=25):
    """Which positions host which residue types, aggregated over all solutions."""
    pair=Counter(); byres=defaultdict(Counter); anch=Counter()
    for s in sols:
        anch[(s['anchor'], s['anchor_resn'])]+=1
        key=[f"{s['anchor_resn']}{s['anchor']}"]
        for h in s['satellites']:
            byres[h['resn']][(h['resi'], h['wt'])]+=1
            key.append(f"{h['resn']}{h['resi']}")
        pair[' + '.join(key)]+=1
    return dict(total=len(sols),
                anchor_positions=[(f'{r}{p}', n) for (p,r),n in anch.most_common(top)],
                satellite_positions={k:[(f'{w}{p}->{k}', n) for (p,w),n in v.most_common(top)]
                                     for k,v in byres.items()},
                combinations=pair.most_common(top))
