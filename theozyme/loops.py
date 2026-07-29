"""Stage B/C: displacement demand, CCD loop closure, conformer generation,
two-sided pocket-preserving scoring."""
import numpy as np
from scipy.spatial import cKDTree
from .geometry import place_atom, rotate_about, ccd_optimal_angle, dihedral

TIERS=[('T0',0.01),('T1',0.80),('T2',2.50),('T3',4.50)]
def tier_of(maxpen):
    for name,cut in TIERS:
        if maxpen<=cut: return name
    return 'T4'

def thermal_envelope(st, resis, chain=None):
    """RMS displacement u = sqrt(B / 8 pi^2) per residue backbone."""
    chain=chain or st.chain[0]; out={}
    for r in resis:
        idx=[i for i in st.residues.get((chain,int(r)),[]) if st.name[i] in ('N','CA','C','O')]
        if not idx: continue
        out[int(r)]=float(np.sqrt(st.b[idx].mean()/(8*np.pi**2)))
    return out

def displacement_demand(st, sub_xyz, mobile_resis, chain=None, clearance=3.40):
    """Per-residue lower bound on the backbone motion needed to clear the substrate."""
    chain=chain or st.chain[0]
    tsub=cKDTree(sub_xyz); per={}
    for r in mobile_resis:
        idx=[i for i in st.residues.get((chain,int(r)),[]) if st.name[i] in ('N','CA','C','O')]
        if not idx: continue
        d,_=tsub.query(st.xyz[idx])
        pen=float(max(0.0, clearance-d.min()))
        if pen>0.01: per[int(r)]=pen
    return per, (max(per.values()) if per else 0.0)

def escape_analysis(st, sub_xyz, per, rigid_tree, chain=None, clearance=3.40):
    """For each residue that must move: direction, whether it is blocked, radial fraction."""
    chain=chain or st.chain[0]; tsub=cKDTree(sub_xyz); cen=sub_xyz.mean(0); out={}
    for r,need in per.items():
        idx=[i for i in st.residues.get((chain,int(r)),[]) if st.name[i] in ('N','CA','C','O')]
        d,j=tsub.query(st.xyz[idx]); k=int(np.argmin(d))
        p=st.xyz[idx[k]]; v=p-sub_xyz[j[k]]; v/=np.linalg.norm(v)
        probe=p+v*need
        blocked=len(rigid_tree.query_ball_point(probe,3.2))
        rad=float(np.dot(v,(p-cen)/np.linalg.norm(p-cen)))
        out[int(r)]=dict(displacement=round(float(need),2), blocked=int(blocked),
                         radial_fraction=round(rad,2), direction=[round(float(x),3) for x in v])
    return out

# ---------------------------------------------------------------- loop building
class LoopSegment:
    """Backbone N/CA/C/O of a contiguous stretch, with fixed anchors either side."""
    def __init__(self, st, start, stop, chain=None, n_anchor=2):
        self.st=st; self.chain=chain or st.chain[0]
        self.start, self.stop = int(start), int(stop)
        self.res=list(range(self.start, self.stop+1))
        self.pre=[r for r in range(self.start-n_anchor, self.start) if (self.chain,r) in st.residues]
        self.post=[r for r in range(self.stop+1, self.stop+1+n_anchor) if (self.chain,r) in st.residues]
        self.idx={}
        for r in self.pre+self.res+self.post:
            for nm in ('N','CA','C','O'):
                for i in st.residues[(self.chain,r)]:
                    if st.name[i]==nm: self.idx[(r,nm)]=i
        self.native=st.xyz.copy()
        self.target=[self.native[self.idx[(self.post[0],nm)]] for nm in ('N','CA','C')]

    def movable(self):
        return [self.idx[(r,nm)] for r in self.res for nm in ('N','CA','C','O') if (r,nm) in self.idx]

    def torsion_axes(self):
        """(origin, axis, downstream-atom-indices) for each phi/psi in the loop."""
        ax=[]
        order=[(r,nm) for r in self.res for nm in ('N','CA','C')]
        for k,(r,nm) in enumerate(order):
            if nm=='N':   o,a=self.idx[(r,'N')],  self.idx[(r,'CA')]      # phi
            elif nm=='CA':o,a=self.idx[(r,'CA')], self.idx[(r,'C')]       # psi
            else: continue
            down=[self.idx[q] for q in order[k+1:]]
            down += [self.idx[(rr,'O')] for rr in self.res if (rr,'O') in self.idx
                     and rr>=r and not (rr==r and nm=='N')]
            ax.append((o,a,sorted(set(down))))
        return ax

    def close(self, xyz, tol=0.10, max_cycles=120):
        """CCD: rotate loop torsions until the post-anchor N/CA/C match their native positions."""
        xyz=xyz.copy(); axes=self.torsion_axes()
        mov=[self.idx[(self.post[0],nm)] for nm in ('N','CA','C')]
        for _ in range(max_cycles):
            rms=np.sqrt(np.mean([np.sum((xyz[m]-t)**2) for m,t in zip(mov,self.target)]))
            if rms<tol: return xyz, rms, True
            for o,a,down in axes:
                th=ccd_optimal_angle([xyz[m] for m in mov], self.target, xyz[o], xyz[a]-xyz[o])
                mv=sorted(set(down)|set(mov))
                xyz[mv]=rotate_about(xyz[mv], xyz[o], xyz[a]-xyz[o], th)
        rms=np.sqrt(np.mean([np.sum((xyz[m]-t)**2) for m,t in zip(mov,self.target)]))
        return xyz, rms, rms<tol

    def perturb(self, xyz, rng, sigma=12.0, n_torsions=4):
        xyz=xyz.copy(); axes=self.torsion_axes()
        if not axes: return xyz
        pick=rng.choice(len(axes), size=min(n_torsions,len(axes)), replace=False)
        for k in pick:
            o,a,down = axes[k]
            th=np.radians(rng.normal(0, sigma))
            xyz[down]=rotate_about(xyz[down], xyz[o], xyz[a]-xyz[o], th)
        return xyz

RAMA_OK=[(-180,-20,-90,50),(-180,-20,90,180),(-160,-40,-70,-10),(40,90,-20,90)]
def rama_fraction(st, seg, xyz):
    ok=tot=0
    for r in seg.res:
        try:
            phi=dihedral(xyz[seg.idx[(r-1,'C')]],xyz[seg.idx[(r,'N')]],
                         xyz[seg.idx[(r,'CA')]],xyz[seg.idx[(r,'C')]])
            psi=dihedral(xyz[seg.idx[(r,'N')]],xyz[seg.idx[(r,'CA')]],
                         xyz[seg.idx[(r,'C')]],xyz[seg.idx[(r+1,'N')]])
        except KeyError: continue
        tot+=1
        if st.resname(r,seg.chain)=='GLY': ok+=1; continue
        if any(a<=phi<=b and c<=psi<=d for a,b,c,d in RAMA_OK): ok+=1
    return ok/max(tot,1)

