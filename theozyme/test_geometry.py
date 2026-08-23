"""Run this FIRST. It is the test that caught a sign error in the NeRF placement
function which had silently mirrored every substrate pose."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from theozyme.geometry import place_atom, dihedral, angle
from theozyme.theozyme import read_xyz

def test_roundtrip(xyz_path):
    el, X, _ = read_xyz(xyz_path)
    quads = [(2,3,4,14),(3,4,14,15),(4,14,15,16),(14,15,16,17),(51,52,53,17)]
    ok = True
    for q in quads:
        a,b,c,d = [X[i] for i in q]
        p = place_atom(a,b,c, np.linalg.norm(d-c), angle(b,c,d), dihedral(a,b,c,d))
        e = float(np.linalg.norm(p-d)); ok &= e < 1e-6
        print(f'  quad {q}: {e:.2e} A')
    return ok

def test_sign(xyz_path):
    """The specific failure: a mirrored torsion still round-trips its own convention,
    so also assert the measured value against a hand-checked reference."""
    el, X, _ = read_xyz(xyz_path)
    t = dihedral(X[4], X[14], X[15], X[16])     # NZ(5),C13(15),C12(16),C11(17) 1-based
    print(f'  torsion_B (NZ=C13-C12-C11) = {t:+.2f} deg  (expect -74.44, NOT +74.44)')
    return abs(t - (-74.44)) < 0.5

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv)>1 else '/mnt/user-data/uploads/Transition_State_Oriented.txt'
    print('round-trip:'); a = test_roundtrip(p)
    print('sign:');       b = test_sign(p)
    print('PASS' if (a and b) else 'FAIL'); sys.exit(0 if (a and b) else 1)
