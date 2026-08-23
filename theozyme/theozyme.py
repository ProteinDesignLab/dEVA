"""Read QM theozyme geometries from XYZ."""
import numpy as np

COV={'H':0.31,'C':0.76,'N':0.71,'O':0.66,'S':1.05,'P':1.07,'F':0.57,'Cl':1.02}
Z2S={1:'H',6:'C',7:'N',8:'O',9:'F',15:'P',16:'S',17:'Cl'}

def read_xyz(path):
    L=open(path).read().split('\n'); n=int(L[0].split()[0]); title=L[1]
    el=[]; xyz=[]
    for l in L[2:2+n]:
        t=l.split(); e=t[0]; e=Z2S[int(e)] if e.isdigit() else e
        el.append(e); xyz.append([float(x) for x in t[1:4]])
    return el, np.array(xyz), title.strip()




