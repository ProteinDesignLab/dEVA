import os
import multiprocessing
from multiprocessing import Pool

import numpy as np

from moleculekit.molecule import Molecule
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering

def create_grid_fromBB(boundingBox, voxelSize=1):
    """Create a grid from a bounding box.

    Parameters
    ----------
    boundingBox : list
        List of the form [xmin, xmax, ymin, ymax, zmin, zmax]
    voxelSize : float
        Size of the voxels in Angstrom

    Returns
    -------
    grid : numpy.ndarray
        Grid of shape (nx, ny, nz)
    box_N : numpy.ndarray
        Number of voxels in each dimension

    """
    # increase grid by 0.5 to sample everything
    xrange = np.arange(boundingBox[0][0], boundingBox[1][0] + 0.5, step=voxelSize)
    yrange = np.arange(boundingBox[0][1], boundingBox[1][1] + 0.5, step=voxelSize)
    zrange = np.arange(boundingBox[0][2], boundingBox[1][2] + 0.5, step=voxelSize)

    gridpoints = np.zeros((xrange.shape[0] * yrange.shape[0] * zrange.shape[0], 3))
    i = 0
    for x in xrange:
        for y in yrange:
            for z in zrange:
                gridpoints[i][0] = x
                gridpoints[i][1] = y
                gridpoints[i][2] = z
                i += 1
    return gridpoints, (xrange.shape[0], yrange.shape[0], zrange.shape[0])


def get_bb(points):
    """Return bounding box from a set of points (N,3)

    Parameters
    ----------
    points : numpy.ndarray
        Set of points (N,3)

    Returns
    -------
    boundingBox : list
        List of the form [xmin, xmax, ymin, ymax, zmin, zmax]

    """
    minx = np.min(points[:, 0])
    maxx = np.max(points[:, 0])

    miny = np.min(points[:, 1])
    maxy = np.max(points[:, 1])

    minz = np.min(points[:, 2])
    maxz = np.max(points[:, 2])
    bb = [[minx, miny, minz], [maxx, maxy, maxz]]
    return bb


def get_all_protein_resids(pdb_file):
    """Return all protein residues from a pdb file

    Parameters
    ----------
    pdb_file : str
        Path to pdb file

    Returns
    -------
    resids : numpy.ndarray
        Array of protein resids old -> new

    """
    try:
        prot = Molecule(pdb_file)
    except:
        exit("could not read file")
    prot.filter("protein and not hydrogen")
    # mapping = prot.renumberResidues(returnMapping=True)
    return prot.get("index", "protein and name CA")


def get_all_metalbinding_resids(pdb_file):
    """Return all metal binding residues from a pdb file

    Parameters
    ----------
    pdb_file : str
        Path to pdb file

    Returns
    -------
    resids : numpy.ndarray
        id of resids that are metal binding

    """
    try:
        prot = Molecule(pdb_file)
    except:
        exit("could not read file")
    prot.renumberResidues()
    prot.filter("protein and not hydrogen")
    return prot.get(
        "index",
        sel="name CA and resname HIS HID HIE HIP CYS CYX GLU GLH GLN ASP ASH ASN GLN MET",
    ), prot.get(
        "resid",
        sel="name CA and resname HIS HID HIE HIP CYS CYX GLU GLH GLN ASP ASH ASN GLN MET",
    )


def compute_average_p_fast(point, cutoff=0.5, k=20, default=1e-5):
    """Using KDTree find the closest gridpoints

    Parameters
    ----------
    point : numpy.ndarray
        Point of shape (3,)
    cutoff : float
        Cutoff distance in Angstrom

    Returns
    -------
    average_p : numpy.ndarray
        Average probability of shape (1,)"""

    d, i = tree.query(point, k=k, distance_upper_bound=cutoff, workers=1)
    m = np.isfinite(d)
    
    return float(output_v[i[m]].mean()) if np.any(m) else float(default)



def get_probability_mean(grid, prot_centers, pvalues):
    """Compute the mean probability of all gridpoints from the globalgrid based on the individual boxes

    Parameters
    ----------
    grid : numpy.ndarray
        Grid of shape (nx, ny, nz)
    prot_centers : numpy.ndarray
        Protein centers of shape (N,3)
    pvalues : numpy.ndarray
        Probability values of shape (N,1)

    Returns
    -------
    mean_p : numpy.ndarray
        Mean probability over grid of shape (nx, ny, nz)
    """
    global output_v
    output_v = pvalues
    global prot_v
    prot_v = prot_centers
    
    try:
        cpuCount = int(os.environ['SLURM_CPUS_ON_NODE'])
    except (KeyError, ValueError):
        cpuCount = multiprocessing.cpu_count()

    global tree
    tree = KDTree(prot_v)
    p = Pool(int(cpuCount/2))
    try:
        results = p.map(compute_average_p_fast, grid)
    finally:
        p.close()
        p.join()
    return np.array(results)


def write_cubefile(bb, pvalues, box_N, outname="Metal3D_pmap.cube", gridres=1):
    """Write a cube file from a probability map
    The cube specification from gaussian is used, distance are converted to bohr

    Parameters
    ----------
    bb : list
        List of the form [xmin, xmax, ymin, ymax, zmin, zmax]
    pvalues : numpy.ndarray
        Probability values of shape (nx, ny, nz)
    box_N : tuple
        Number of voxels in each dimension
    outname : str
        Name of the output file
    gridres:float
        Resolution of the grid used for writing the voxels

    """

    with open(outname, "w") as cube:
        cube.write(" Metal3D Cube File\n")
        cube.write(" Outer Loop: X, Middle Loop y, inner Loop z\n")

        angstromToBohr = 1.89
        cube.write(
            f"    1   {bb[0][0]*angstromToBohr: .6f}  {bb[0][1]*angstromToBohr: .6f}   {bb[0][2]*angstromToBohr: .6f}\n"
        )
        cube.write(
            f"{str(box_N[0]).rjust(5)}    {1.890000*gridres:.9f}    0.000000    0.000000\n"
        )
        cube.write(
            f"{str(box_N[1]).rjust(5)}    0.000000    {1.890000*gridres:.9f}    0.000000\n"
        )
        cube.write(
            f"{str(box_N[2]).rjust(5)}    0.000000    0.000000    {1.890000*gridres:.9f}\n"
        )
        cube.write("    1    1.000000    0.000000    0.000000    0.000000\n")

        o = pvalues.reshape(box_N)
        for x in range(box_N[0]):
            for y in range(box_N[1]):
                for z in range(box_N[2]):
                    cube.write(f" {o[x][y][z]: .5E}")
                    if z % 6 == 5:
                        cube.write("\n")
                cube.write("\n")


import warnings


def find_unique_sites(
    pvalues, grid, writeprobes=False, probefile="probes.pdb", threshold=7, p=0.0001
):
    """The probability voxels are points and the voxel clouds may contain multiple metals
    This function finds the unique sites and returns the coordinates of the unique sites.
    It uses the AgglomerativeClustering algorithm to find the unique sites.
    The threshold is the maximum distance between two points in the same cluster it can be changed to get more metal points.
    """

    points = grid[pvalues > p]
    point_p = pvalues[pvalues > p]
    if len(points) == 0:
        return None
    if len(points) == 1:
        sites = [[list(points[0]), float(point_p[0])]]
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, linkage="complete", distance_threshold=threshold
        ).fit(points)

        sites = []
        for i in range(clustering.n_clusters_):
            c_points = points[clustering.labels_ == i]
            c_points_p = point_p[clustering.labels_ == i]

            # compute center of probabilities as COM
            CM = np.average(c_points, axis=0, weights=c_points_p)
            position = list(CM)
            cp = float(np.max(c_points_p))
            sites.append([position, cp])

    if writeprobes:
        with open(probefile, "w") as f:
            for i, site in enumerate(sites):
                f.write(
                    f"HETATM  {i+1:3} ZN    ZN A  {1+i}    {site[0][0]: 8.3f}{site[0][1]: 8.3f}{site[0][2]: 8.3f}  {site[1]:.2f}  0.0           ZN2+\n"
                )
    return sites


def maxprobability(pvalues, grid, pdb, label):
    """Returns the point in the grid with the highest probability"""

    if os.path.isfile("maxp_" + os.path.basename(label) + ".csv"):
        with open("maxp_" + os.path.basename(label) + ".csv", "a") as fp:
            fp.write(
                f"{pdb},{np.max(pvalues):.4f}, {grid[np.argmax(pvalues)][0]: .4f}, {grid[np.argmax(pvalues)][1]: .4f}, {grid[np.argmax(pvalues)][2]: .4f}\n"
            )
    else:
        with open("maxp_" + os.path.basename(label) + ".csv", "w") as fp:
            fp.write("pdb,p,x,y,z\n")
            fp.write(
                f"{pdb},{np.max(pvalues):.4f}, {grid[np.argmax(pvalues)][0]: .4f}, {grid[np.argmax(pvalues)][1]: .4f}, {grid[np.argmax(pvalues)][2]: .4f}\n"
            )