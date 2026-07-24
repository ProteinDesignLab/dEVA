import copy

from .data_utils import (
    featurize,
    write_full_PDB,
)

from .sc_utils import pack_side_chains

def pack_sc(args, model, protein_dict, S, other_atoms, icodes, outfile, device="cuda"):
    #print("Packing side chains...")
    
    feature_dict_ = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=1,
        number_of_ligand_atoms=16,
        model_type="ligand_mpnn",
    )

    sc_feature_dict = copy.deepcopy(feature_dict_)
    B = 1

    for k, v in sc_feature_dict.items():
        if k != "S":
            try:
                num_dim = len(v.shape)
                if num_dim == 2:
                    sc_feature_dict[k] = v.repeat(B, 1)
                elif num_dim == 3:
                    sc_feature_dict[k] = v.repeat(B, 1, 1)
                elif num_dim == 4:
                    sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                elif num_dim == 5:
                    sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
            except:
                pass

    if len(S.shape) == 1:
        S = S.to(device).unsqueeze(-1).unsqueeze(0)
    if len(S.shape) == 3:
        S = S.to(device).squeeze(-1)

    sc_feature_dict["S"] = S.to(device)
    
    model.to(device)
    sc_dict = pack_side_chains(
        sc_feature_dict,
        model,
        args.sc_num_denoising_steps,
        args.sc_num_samples,
        args.repack_everything,
    )
    model.cpu()

    X_list = sc_dict["X"][0]
    X_m_list = sc_dict["X_m"][0]
    b_factor_list = sc_dict["b_factors"][0]
    if other_atoms: 
        write_full_PDB(
            outfile,
            X_list.cpu().numpy(),
            X_m_list.cpu().numpy(),
            b_factor_list.detach().cpu().numpy(),
            feature_dict_["R_idx_original"][0].cpu().numpy(),
            protein_dict["chain_letters"],
            S.squeeze().cpu().numpy(),
            other_atoms=other_atoms,
            icodes=icodes,
            force_hetatm=1,
        )
    else:
        write_full_PDB(
            outfile,
            X_list.cpu().numpy(),
            X_m_list.cpu().numpy(),
            b_factor_list.detach().cpu().numpy(),
            feature_dict_["R_idx_original"][0].cpu().numpy(),
            protein_dict["chain_letters"],
            S.squeeze().cpu().numpy(),
            icodes=icodes,
            force_hetatm=1,
        )
def add_header(filename, fitnesses: dict, sequence: str):
    """Write a standardized REMARK header with all fitness metrics."""
    # Dynamically build REMARK lines for every fitness key/value
    fitness_lines = "\n".join(
        [f"REMARK 220 REMARK: {key} = {value}" for key, value in fitnesses.items()]
    )

    header = (
        f"{fitness_lines}\n"
        f"REMARK 220 REMARK: sequence = {sequence}\n"
        "REMARK 220 VERSION EVOLVE-1\n"
    )
    # write to beginning of pdb file
    with open(filename) as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        f.write(header)
        f.writelines(lines)