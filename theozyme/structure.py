"""Minimal PDB container. No external dependencies."""
import numpy as np

AA3to1 = dict(ALA='A',ARG='R',ASN='N',ASP='D',CYS='C',GLN='Q',GLU='E',GLY='G',HIS='H',
              ILE='I',LEU='L',LYS='K',MET='M',PHE='F',PRO='P',SER='S',THR='T',TRP='W',
              TYR='Y',VAL='V')
BACKBONE = ('N','CA','C','O')

class Structure:
    def __init__(self, path=None):
        self.rec=[]; self.name=[]; self.resn=[]; self.chain=[]; self.resi=[]
        self.occ=[]; self.b=[]; self.elem=[]; xyz=[]
        if path:
            for l in open(path):
                if not l.startswith(('ATOM','HETATM')): continue
                if l[16] not in (' ','A'): continue          # first altloc only
                self.rec.append(l[:6].strip()); self.name.append(l[12:16].strip())
                self.resn.append(l[17:20].strip()); self.chain.append(l[21])
                self.resi.append(int(l[22:26])); self.occ.append(float(l[54:60] or 1))
                self.b.append(float(l[60:66] or 0)); self.elem.append(l[76:78].strip() or l[12:14].strip()[0])
                xyz.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        self.xyz=np.array(xyz,float)
        for k in ('rec','name','resn','chain','resi','occ','b','elem'):
            setattr(self,k,np.array(getattr(self,k)))
        self._index()

    def _index(self):
        self.residues={}
        for i,(c,r) in enumerate(zip(self.chain,self.resi)):
            self.residues.setdefault((c,int(r)),[]).append(i)
        self.protein_res=[k for k,v in self.residues.items() if self.rec[v[0]]=='ATOM']
        self.protein_res.sort(key=lambda k:(k[0],k[1]))

    def atom(self, resi, name, chain=None):
        chain = chain or self.chain[0]
        for i in self.residues.get((chain,int(resi)),[]):
            if self.name[i]==name: return self.xyz[i]
        return None

    def resname(self, resi, chain=None):
        chain = chain or self.chain[0]
        v=self.residues.get((chain,int(resi)))
        return self.resn[v[0]] if v else None

    def sequence(self):
        return ''.join(AA3to1.get(self.resn[self.residues[k][0]],'X') for k in self.protein_res)

    def select(self, rec=None, names=None, resis=None, exclude_resis=None):
        m=np.ones(len(self.xyz),bool)
        if rec is not None: m &= (self.rec==rec)
        if names is not None: m &= np.isin(self.name,list(names))
        if resis is not None: m &= np.isin(self.resi,list(resis))
        if exclude_resis is not None: m &= ~np.isin(self.resi,list(exclude_resis))
        return np.where(m)[0]

    def backbone_idx(self, include_cb=True, exclude_resis=None):
        names = BACKBONE + (('CB',) if include_cb else ())
        return self.select(rec='ATOM', names=names, exclude_resis=exclude_resis)

    def ligand_res(self):
        return [k for k,v in self.residues.items() if self.rec[v[0]]=='HETATM']

    def append_atoms(self, atoms):
        """Append atom records in-place. Each item is a dict with keys
        rec, name, resn, chain, resi, xyz, and optional occ/b/elem."""
        if not atoms:
            return self
        rec, name, resn, chain, resi, occ, b, elem, xyz = (
            list(self.rec), list(self.name), list(self.resn), list(self.chain),
            list(self.resi), list(self.occ), list(self.b), list(self.elem),
            list(self.xyz))
        for a in atoms:
            rec.append(str(a.get('rec', 'ATOM')))
            name.append(str(a['name']))
            resn.append(str(a['resn']))
            chain.append(str(a['chain']))
            resi.append(int(a['resi']))
            occ.append(float(a.get('occ', 1.0)))
            b.append(float(a.get('b', 0.0)))
            nm = str(a['name'])
            elem.append(str(a.get('elem') or (nm[0] if nm else 'X')))
            xyz.append(np.asarray(a['xyz'], float).reshape(3))
        self.rec = np.array(rec)
        self.name = np.array(name)
        self.resn = np.array(resn)
        self.chain = np.array(chain)
        self.resi = np.array(resi, int)
        self.occ = np.array(occ, float)
        self.b = np.array(b, float)
        self.elem = np.array(elem)
        self.xyz = np.asarray(xyz, float)
        self._index()
        return self

    @staticmethod
    def _fmt(rec, serial, name, resn, chain, resi, xyz, occ, b, elem):
        """PDB v3.3 fixed columns. Atom names with a one-character element symbol are
        indented by one (cols 14-16); four-character names start at column 13."""
        nm = name if len(name) >= 4 else ' ' + name.ljust(3)
        x, y, z = xyz
        return (f'{rec:<6s}{serial:5d} {nm:4s} {resn:>3s} {chain:1s}{resi:4d}    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {elem:>2s}')

    def write(self, path, extra=None, remarks=(), conect=None, links=()):
        L=[f'REMARK  {r}' for r in remarks]; n=0
        self.serial={}
        for i in range(len(self.xyz)):
            n+=1
            self.serial[(str(self.chain[i]), int(self.resi[i]), str(self.name[i]))] = n
            L.append(self._fmt(self.rec[i], n, self.name[i], self.resn[i], self.chain[i],
                               int(self.resi[i]), self.xyz[i], 1.00, self.b[i], self.elem[i]))
        for e in (extra or []):
            n+=1
            self.serial[(str(e.get('chain','X')), int(e['resi']), str(e['name']))] = n
            L.append(self._fmt('HETATM', n, e['name'], e['resn'], e.get('chain','X'),
                               int(e['resi']), e['xyz'], 1.00, 0.00, e['elem']))
        for lk in links:
            a, b = lk['a'], lk['b']
            L.append(f"LINK        {a['name']:<4s} {a['resn']:>3s} {a['chain']}{a['resi']:4d}"
                     f"{'':16s}{b['name']:<4s} {b['resn']:>3s} {b['chain']}{b['resi']:4d}"
                     f"  1555   1555 {lk['dist']:5.2f}")
        if conect:
            for a in sorted(conect):
                bs = sorted(conect[a])
                for k in range(0, len(bs), 4):
                    L.append(f'CONECT{a:5d}' + ''.join(f'{x:5d}' for x in bs[k:k+4]))
        open(path,'w').write('\n'.join(L)+'\nEND\n')

def assign_helices(st, chain=None):
    """DSSP-style backbone H-bond energy; returns set of helical residue numbers."""
    chain = chain or st.chain[0]
    nums=[r for c,r in st.protein_res if c==chain]
    N={r:st.atom(r,'N',chain) for r in nums}; CA={r:st.atom(r,'CA',chain) for r in nums}
    C={r:st.atom(r,'C',chain) for r in nums}; O={r:st.atom(r,'O',chain) for r in nums}
    H={}
    for i,r in enumerate(nums):
        if i==0 or st.resname(r,chain)=='PRO': continue
        p=nums[i-1]
        if p!=r-1 or C[p] is None or O[p] is None or N[r] is None: continue
        v=C[p]-O[p]; H[r]=N[r]+v/np.linalg.norm(v)
    def E(i,j):
        if j not in H or O.get(i) is None: return 0
        q=0.42*0.20*332
        return q*(1/np.linalg.norm(O[i]-N[j]) + 1/np.linalg.norm(C[i]-H[j])
                  - 1/np.linalg.norm(O[i]-H[j]) - 1/np.linalg.norm(C[i]-N[j]))
    hb={(i,j) for i in nums for j in nums
        if abs(i-j)>2 and j in H and np.linalg.norm(CA[i]-CA[j])<9.0 and E(i,j)<-0.5}
    hel=set()
    for r in nums:
        if (r,r+4) in hb and (r+1,r+5) in hb: hel.update(range(r+1,r+5))
    return hel

def assign_strands(st, chain=None, min_len=3):
    """Beta strands via DSSP bridge criteria. Includes the PARALLEL rule, which a
    TIM barrel needs -- an antiparallel-only test finds nothing in a (ba)8 fold."""
    chain = chain or st.chain[0]
    nums=[r for c,r in st.protein_res if c==chain]
    N={r:st.atom(r,'N',chain) for r in nums}; CA={r:st.atom(r,'CA',chain) for r in nums}
    C={r:st.atom(r,'C',chain) for r in nums}; O={r:st.atom(r,'O',chain) for r in nums}
    H={}
    for i,r in enumerate(nums):
        if i==0 or st.resname(r,chain)=='PRO': continue
        p=nums[i-1]
        if p!=r-1 or C[p] is None or O[p] is None or N[r] is None: continue
        v=C[p]-O[p]; H[r]=N[r]+v/np.linalg.norm(v)
    def E(i,j):
        if j not in H or O.get(i) is None: return 0
        q=0.42*0.20*332
        return q*(1/np.linalg.norm(O[i]-N[j])+1/np.linalg.norm(C[i]-H[j])
                  -1/np.linalg.norm(O[i]-H[j])-1/np.linalg.norm(C[i]-N[j]))
    hb={(i,j) for i in nums for j in nums
        if abs(i-j)>2 and j in H and np.linalg.norm(CA[i]-CA[j])<9.0 and E(i,j)<-0.5}
    res=set()
    for i in nums:
        for j in nums:
            if j<=i+2: continue
            par  = ((i-1,j) in hb and (j,i+1) in hb) or ((j-1,i) in hb and (i,j+1) in hb)
            anti = ((i,j) in hb and (j,i) in hb) or ((i-1,j+1) in hb and (j-1,i+1) in hb)
            if par or anti: res |= {i,j}
    segs=[]; cur=[]
    for r in nums:
        if r in res: cur.append(r)
        else:
            if len(cur)>=min_len: segs.append((cur[0],cur[-1]))
            cur=[]
    if len(cur)>=min_len: segs.append((cur[0],cur[-1]))
    return segs

_WATER = frozenset(('HOH', 'WAT', 'DOD', 'H2O'))

def organic_ligand_keys(st):
    """Non-water HETATM residue keys, largest first (real ligands over ions)."""
    keys = [k for k in st.ligand_res() if str(st.resn[st.residues[k][0]]) not in _WATER]
    keys.sort(key=lambda k: -len(st.residues[k]))
    return keys

def pocket_center(st, chain=None):
    """Active-site reference for shell / toward filters.

    Prefer a non-water ligand centroid when present. On an empty TIM barrel,
    use the mean of beta-strand C-terminal CAs (catalytic mouth), not the
    whole-protein backbone centroid — mid-barrel centers flip the CA→CB
    'toward' sign for mouth residues (e.g. 4A29 Thr83).
    """
    chain = chain or (str(st.chain[0]) if len(st.chain) else 'A')
    lig = organic_ligand_keys(st)
    if lig:
        return st.xyz[st.residues[lig[0]]].mean(0), f'ligand:{st.resn[st.residues[lig[0]][0]]}{lig[0][1]}'
    segs = assign_strands(st, chain)
    cas = []
    for a, b in segs:
        ca = st.atom(b, 'CA', chain)
        if ca is not None:
            cas.append(ca)
    if len(cas) >= 3:
        return np.mean(cas, axis=0), 'strand_C_mouth'
    return st.xyz[st.backbone_idx()].mean(0), 'backbone'

def barrel_shell(st, center, chain=None, radius=13.0, min_len=3, require_toward=0.0):
    """Beta-strand positions whose CB is within radius of center and CA→CB faces it."""
    chain = chain or st.chain[0]
    segs = assign_strands(st, chain, min_len)
    out=[]
    for a,b in segs:
        for r in range(a,b+1):
            ca,cb = st.atom(r,'CA',chain), st.atom(r,'CB',chain)
            if cb is None: continue
            d=float(np.linalg.norm(cb-center))
            if d>radius: continue
            v1,v2 = cb-ca, center-cb
            toward=float(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
            if toward < require_toward: continue
            out.append(dict(resi=int(r), wt=str(st.resname(r,chain)), strand=f'{a}-{b}',
                            cb_pocket=round(d,1), toward=round(toward,2)))
    return sorted(out, key=lambda x:x['cb_pocket']), segs
