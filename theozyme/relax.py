"""Stage D: restrained Cartesian relaxation.

Elastic-network + steric-repulsion + positional restraints, with region-dependent
stiffness so the barrel is held tightly and the loops are free. Analytic gradients,
L-BFGS. This is a geometry regulariser and clash reliever, not a force field --
it composes with a real FastRelax / OpenMM minimisation afterwards.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

class RestrainedRelax:
    def __init__(self, st, mobile_resis, substrate_xyz=None, chain=None,
                 enm_cutoff=9.0, k_rigid=8.0, k_mobile=0.6, k_pos_rigid=2.0,
                 k_pos_mobile=0.02, rep_dmin=3.30, k_rep=25.0, sub_dmin=3.40):
        self.st=st; self.chain=chain or st.chain[0]
        self.mobile=set(int(r) for r in mobile_resis)
        idx=st.backbone_idx(include_cb=True)
        self.idx=idx; self.x0=st.xyz[idx].copy(); self.resi=st.resi[idx]
        self.is_mob=np.isin(self.resi,list(self.mobile))
        n=len(idx)
        # elastic network from the native structure
        t=cKDTree(self.x0); pairs=np.array(sorted(t.query_pairs(enm_cutoff)))
        self.pi,self.pj=pairs[:,0],pairs[:,1]
        self.d0=np.linalg.norm(self.x0[self.pi]-self.x0[self.pj],axis=1)
        soft=self.is_mob[self.pi]|self.is_mob[self.pj]
        self.kenm=np.where(soft,k_mobile,k_rigid)
        # bonded pairs (same or adjacent residue, short) get full stiffness regardless
        bonded=(np.abs(self.resi[self.pi]-self.resi[self.pj])<=1)&(self.d0<2.0)
        self.kenm=np.where(bonded,max(k_rigid,20.0),self.kenm)
        self.bonded=bonded
        self._kposm, self._kposr = k_pos_mobile, k_pos_rigid
        self.kpos=np.where(self.is_mob,k_pos_mobile,k_pos_rigid)
        self.x0_current=None
        self.rep_dmin=rep_dmin; self.k_rep=k_rep; self.sub_dmin=sub_dmin
        self.sub=substrate_xyz
        self.extra=[]      # flat-bottom distance restraints: (i, target_xyz, lo, hi, k)
        self._pp=None; self._sp=None; self.pad=1.2

    def _build_pairs(self, X):
        t=cKDTree(X)
        pr=np.array(sorted(t.query_pairs(self.rep_dmin+self.pad)) or [], dtype=int)
        if len(pr):
            pr=pr[np.abs(self.resi[pr[:,0]]-self.resi[pr[:,1]])>1]
        self._pp=pr
        if self.sub is not None:
            ts=cKDTree(self.sub)
            nb=ts.query_ball_point(X, self.sub_dmin+self.pad)
            sp=[(i,j) for i,js in enumerate(nb) for j in js]
            self._sp=np.array(sp,dtype=int) if sp else np.zeros((0,2),int)

    def add_reach(self, resi, target_xyz, lo, hi, k=20.0, name='CB'):
        for i,j in enumerate(self.idx):
            if self.st.resi[j]==resi and self.st.name[j]==name:
                self.extra.append((i,np.asarray(target_xyz,float),lo,hi,k)); return True
        return False

    def _energy(self, x):
        X=x.reshape(-1,3); E=0.0; G=np.zeros_like(X)
        v=X[self.pi]-X[self.pj]; d=np.linalg.norm(v,axis=1)
        dd=d-self.d0; E+=np.sum(self.kenm*dd**2)
        g=(2*self.kenm*dd/np.maximum(d,1e-8))[:,None]*v
        np.add.at(G,self.pi,g); np.add.at(G,self.pj,-g)
        dx=X-self.x0; E+=np.sum(self.kpos[:,None]*dx**2); G+=2*self.kpos[:,None]*dx
        # steric repulsion on a cached, padded pair list (rebuilt between stages)
        if self._pp is not None and len(self._pp):
            v=X[self._pp[:,0]]-X[self._pp[:,1]]; d=np.linalg.norm(v,axis=1)
            act=d<self.rep_dmin
            if act.any():
                pa=self._pp[act]; va=v[act]; da=d[act]; ov=self.rep_dmin-da
                E+=self.k_rep*np.sum(ov**2)
                g=(-2*self.k_rep*ov/np.maximum(da,1e-8))[:,None]*va
                np.add.at(G,pa[:,0],g); np.add.at(G,pa[:,1],-g)
        if self._sp is not None and len(self._sp):
            pi_,sj = self._sp[:,0], self._sp[:,1]
            v=X[pi_]-self.sub[sj]; d=np.linalg.norm(v,axis=1)
            act=d<self.sub_dmin
            if act.any():
                ov=self.sub_dmin-d[act]
                E+=self.k_rep*np.sum(ov**2)
                g=(-2*self.k_rep*ov/np.maximum(d[act],1e-8))[:,None]*v[act]
                np.add.at(G,pi_[act],g)
        for i,tgt,lo,hi,k in self.extra:
            v=X[i]-tgt; d=np.linalg.norm(v)
            if d<lo: e=lo-d
            elif d>hi: e=d-hi
            else: continue
            E+=k*e**2
            s=-1 if d<lo else 1
            G[i]+= 2*k*e*s*v/max(d,1e-8)
        return E, G.ravel()

    def run_staged(self, schedule=((1.0,1.0),(2.5,0.4),(6.0,0.15)), maxiter=300,
                   target_clearance=None):
        """Ramp steric repulsion up while relaxing the positional restraints on the
        mobile region. Stops early once the clearance target is met."""
        k_rep0, kposm0 = self.k_rep, None
        best=None
        for f_rep, f_pos in schedule:
            self.k_rep = k_rep0*f_rep
            self.kpos = np.where(self.is_mob, self._kposm*f_pos, self._kposr)
            r=self.run(maxiter=maxiter)
            self.x0_current=r['xyz'][self.idx]
            best=r
            if target_clearance is not None:
                d,_=cKDTree(self.sub).query(r['xyz'][self.idx])
                if d.min()>=target_clearance: break
        self.k_rep=k_rep0
        return best

    def run(self, maxiter=400):
        start = self.x0_current if self.x0_current is not None else self.x0
        self._build_pairs(start)
        r=minimize(self._energy, start.ravel(), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, maxfun=maxiter*2))
        X=r.x.reshape(-1,3)
        disp=np.linalg.norm(X-self.x0,axis=1)
        out=self.st.xyz.copy(); out[self.idx]=X
        return dict(xyz=out, energy=float(r.fun), niter=int(r.nit),
                    max_disp_mobile=float(disp[self.is_mob].max()) if self.is_mob.any() else 0.0,
                    rmsd_mobile=float(np.sqrt((disp[self.is_mob]**2).mean())) if self.is_mob.any() else 0.0,
                    max_disp_rigid=float(disp[~self.is_mob].max()),
                    rmsd_rigid=float(np.sqrt((disp[~self.is_mob]**2).mean())))

def propagate_sidechains(st, new_xyz, chain=None):
    """Backbone-only relaxation moves N/CA/C/O/CB. Carry each residue's remaining
    side-chain atoms along with the rigid transform of its own N-CA-C frame, so the
    output structure is chemically intact."""
    from .geometry import kabsch
    chain=chain or st.chain[0]
    out=new_xyz.copy()
    for (c,r),idx in st.residues.items():
        if st.rec[idx[0]]!='ATOM': continue
        core=[i for i in idx if st.name[i] in ('N','CA','C')]
        rest=[i for i in idx if st.name[i] not in ('N','CA','C','O','CB')]
        if len(core)<3 or not rest: continue
        R,t=kabsch(st.xyz[core], new_xyz[core])
        out[rest]=(R@st.xyz[rest].T).T+t
    return out
