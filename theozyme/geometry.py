"""Geometric primitives: NeRF placement, dihedrals, Kabsch, CCD loop closure, backrub."""
import numpy as np

def norm(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)

def place_atom(a, b, c, dist, ang, tor):
    """NeRF. Place D given A-B-C, |CD|=dist, angle(B,C,D)=ang, torsion(A,B,C,D)=tor.
    a,b,c may be (3,) or (M,3); tor may be scalar or (M,)."""
    a, b, c = np.atleast_2d(a), np.atleast_2d(b), np.atleast_2d(c)
    ang = np.radians(ang); tor = np.radians(np.atleast_1d(np.asarray(tor, float)))
    bc = norm(c - b)
    n = norm(np.cross(b - a, bc))
    m = np.cross(n, bc)
    d = (c + (-dist * np.cos(ang)) * bc
           + (dist * np.sin(ang) * np.cos(tor))[:, None] * m
           + (dist * np.sin(ang) * np.sin(tor))[:, None] * n)
    return d[0] if d.shape[0] == 1 else d

def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))

def angle(p0, p1, p2):
    v1, v2 = p0 - p1, p2 - p1
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def kabsch(P, Q):
    """Rotation+translation mapping P onto Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc

def rotmat(axis, theta):
    """Rodrigues rotation matrix, theta in radians."""
    u = axis / np.linalg.norm(axis)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def rotate_about(coords, origin, axis, theta):
    R = rotmat(axis, theta)
    return (R @ (coords - origin).T).T + origin

def ccd_optimal_angle(moving, targets, origin, axis):
    """Analytic CCD step (Canutescu & Dunbrack 2003).
    Returns the rotation angle about (origin, axis) minimising sum |target - moving|^2."""
    u = axis / np.linalg.norm(axis)
    A = B = 0.0
    for M, F in zip(moving, targets):
        O = origin + np.dot(M - origin, u) * u
        rvec = M - O
        r = np.linalg.norm(rvec)
        if r < 1e-6:
            continue
        rhat = rvec / r
        shat = np.cross(u, rhat)
        f = F - O
        A += r * np.dot(f, rhat)
        B += r * np.dot(f, shat)
    return np.arctan2(B, A)

