
import parmed
import numpy as np
import torch
import warnings
import os
import gc

from .utils.voxelization import processStructures as processStructures
from .utils.helpers import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def run(model, 
        output_metals=".", 
        output_gen=".", 
        pdb='curr.pdb', 
        pdb_name = 'curr', 
        var_idx='', 
        max_metal_p=0.2, 
        overwrite=True, 
        threshold=7,
        device='cuda'):

    if len(var_idx) > 0:
        mol = Molecule(pdb)
        mol.filter("protein and not hydrogen")
        ids = mol.get("index", f"resid {var_idx} and name CA")
        resnames = mol.get("resname", f"resid {var_idx} and name CA")
        #resid = var_idx.split(" ")
    else:
        ids, _ = get_all_metalbinding_resids(pdb)
        resnames = []
    
    voxels, prot_centers, prot_N, prots = processStructures(pdb, ids)
    
    # Pre-allocate outputs tensor on CPU to avoid repeated allocations
    outputs = torch.zeros([voxels.size()[0], 1, 32, 32, 32], device='cpu')
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            for i in range(0, voxels.size()[0], 1):
                outputs[i:i+1] = model(voxels[i:i+1].to(device)).detach().cpu()

    # More efficient tensor operations
    t = outputs.view(outputs.shape[0], 32 * 32 * 32)
    per_res_p, _ = torch.max(t, 1)
    
    # process all predicted probabilities
    prot_v = np.vstack(prot_centers)
    output_v = outputs.flatten().numpy()
    
    bb = get_bb(prot_v)
    grid, _ = create_grid_fromBB(bb)
    probability_values = get_probability_mean(grid, prot_v, output_v)

    result = find_unique_sites(
        probability_values,
        grid,
        writeprobes=False,  # Don't write all probes initially
        probefile=f'{output_metals}.pdb',
        threshold=threshold,
        p=max_metal_p,
    )

    with open(f"{output_metals}.results", "w") as f:
        f.write(str(list(per_res_p.numpy())))
        f.write(str(np.max(probability_values).item()))

    m3d = None
    found = False
    max_idx = -1
    if result is not None:
        _p = [l[1] for l in result]
        max_site = max(result, key=lambda x: x[1])
        max_idx, max_site = max(enumerate(result), key=lambda t: t[1][1])  
        
        # Write only the maximum probability zinc to the probe file
        with open(f'{output_metals}.pdb', "w") as f:
            f.write(f"HETATM  {1:3} ZN    ZN A  {1}    {max_site[0][0]: 8.3f}{max_site[0][1]: 8.3f}{max_site[0][2]: 8.3f}  {max_site[1]:.2f}  0.0           ZN2+\n")
        
        m3d = {"p": _p[max_idx],
               "per_res_p": list(per_res_p.numpy()),
               "resnames": list(resnames),
               "probefile": f'{output_metals}.pdb'}
        
        found = True
        max_idx = 0

    m3d = None
    found = False

    if result is not None:
        probs = [l[1] for l in result]
        topX = sorted(enumerate(result), key=lambda t: t[1][1], reverse=True)[:1] # X = 1

        with open(f'{output_metals}.pdb', "w") as f:
            for i, (idx, (coords, p)) in enumerate(topX, start=1):
                x, y, z = coords
                f.write(
                    f"HETATM{i:5d} ZN    ZN A{i:4d}"
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{p:6.2f}  0.0           ZN2+\n"
                )

        m3d = {
            "p": [p for (_, (_, p)) in topX],  # top X probabilities
            "per_res_p": list(per_res_p.numpy()),
            "resnames": list(resnames),
            "probefile": f"{output_metals}.pdb"
        }

    found = True

    p1 = parmed.load_file(pdb)
    outfile = f'{output_gen}_metal3d.pdb'

    if m3d == None:
        metal_predicted = 0.01
        p1.save(outfile, overwrite=overwrite)
        return metal_predicted, outfile, False
    else:
        metal_predicted = float(sum(m3d["p"]))/len(m3d["p"]) # write out sum
        if metal_predicted > max_metal_p:
            p2 = parmed.load_file(m3d["probefile"])
            combined = p1['!@ZN'] + p2
            combined.save(outfile, overwrite=overwrite)
        else:
            p1.save(outfile, overwrite=overwrite)
        return metal_predicted, outfile, found
