import torch
import json
import numpy as np

from .data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    restype_int_to_str,
    restype_str_to_int,
)

def get_var_residues(args, pdb):
    var_residues = [item for item in args.var_residues.split()]

    return var_residues

def get_fixed_residues(args, pdb):
    fixed_residues = [item for item in args.fixed_residues.split()]

    return fixed_residues

def get_bias_aa(args, pdb, device='cuda'):
    # Get biases for amino acid types
    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_str_to_int[AA]] = a2[i]
    bias_AA_per_residue = {}
    
    if args.bias_AA_per_residue:
        with open(args.bias_AA_per_residue, "r") as fh:
            bias_AA_per_residue = json.load(fh)[pdb]  # {"A12": {"G": 1.1}}
    
    return bias_AA, bias_AA_per_residue

def get_omit_aa(args, pdb, device='cuda'):
    # Omit amino acids 
    omit_AA_per_residue = {}
    if args.omit_AA_per_residue:
        with open(args.omit_AA_per_residue, "r") as fh:
            omit_AA_per_residue = json.load(fh)[pdb]  # {"A12": "PG"}
    
    omit_AA_list = args.omit_AA
    omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
        device=device)
    
    return omit_AA, omit_AA_per_residue

def get_parse_chains(args, pdb_paths):
    if len(args.parse_these_chains_only) != 0:
        parse_these_chains_only_list = args.parse_these_chains_only.split(",")
    else:
        parse_these_chains_only_list = []

    return parse_these_chains_only_list

def get_encoded_residues(protein_dict, icodes):

    encoded_residues = []
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
    chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
    
    for i, R_idx_item in enumerate(R_idx_list):
        tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
        encoded_residues.append(tmp)
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    encoded_residue_dict_rev = dict(zip(list(range(len(encoded_residues))), encoded_residues))
    
    return encoded_residues, encoded_residue_dict, encoded_residue_dict_rev

# fix/streamline
def bias_aa(args, encoded_residues, bias_AA_per_residue_multi, device='cuda'):
    bias_AA_per_residue = torch.zeros([len(encoded_residues), 21], device=device, dtype=torch.float32)
    
    if args.bias_AA_per_residue:
        bias_dict = bias_AA_per_residue_multi[pdb]
        for residue_name, v1 in bias_dict.items():
            if residue_name in encoded_residues:
                i1 = encoded_residue_dict[residue_name]
                for amino_acid, v2 in v1.items():
                    if amino_acid in alphabet:
                        j1 = restype_str_to_int[amino_acid]
                        bias_AA_per_residue[i1, j1] = v2
    return bias_AA_per_residue

# fix/streamline
def omit_aa(args, encoded_residues, encoded_residue_dict, omit_AA_per_residue_multi, device='cuda'):
    omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
    if args.omit_AA_per_residue:
        omit_dict = omit_AA_per_residue_multi[pdb]
        for residue_name, v1 in omit_dict.items():
            if residue_name in encoded_residues:
                i1 = encoded_residue_dict[residue_name]
                for amino_acid in v1:
                    if amino_acid in alphabet:
                        j1 = restype_str_to_int[amino_acid]
                        omit_AA_per_residue[i1, j1] = 1.0
    
    return omit_AA_per_residue

def get_symmetry(args, encoded_residue_dict):
    # specify which residues are linked
    if args.symmetry_residues:
        symmetry_residues_list_of_lists = [
            x.split(",") for x in args.symmetry_residues.split("|")
        ]
        remapped_symmetry_residues = []
        for t_list in symmetry_residues_list_of_lists:
            tmp_list = []
            for t in t_list:
                tmp_list.append(encoded_residue_dict[t])
            remapped_symmetry_residues.append(tmp_list)
    else:
        remapped_symmetry_residues = [[]]

    # specify linking weights
    if args.symmetry_weights:
        symmetry_weights = [
            [float(item) for item in x.split(",")]
            for x in args.symmetry_weights.split("|")
        ]
    else:
        symmetry_weights = [[]]

    return remapped_symmetry_residues, symmetry_weights

def prepare_ligandmpnn(args, 
                       protein_dict, 
                       icodes, 
                       design_params, 
                       atom_context_num, 
                       use_atom_context=1, 
                       device='cuda'):

    encoded_residues, encoded_residue_dict, encoded_residue_dict_rev = get_encoded_residues(protein_dict, icodes)
    bias_AA_per_residue = bias_aa(args, encoded_residues, design_params['bias_AA_per_residue'], device=device)
    omit_AA_per_residue = omit_aa(args, encoded_residues, encoded_residue_dict, design_params['omit_AA_per_residue'], device=device)

    fixed_residues = design_params['fixed_residues']
    var_residues = design_params['var_residues']

    if len(args.chains_to_design) != 0:
        chains_to_design_list = args.chains_to_design.split(",")
    else:
        chains_to_design_list = protein_dict["chain_letters"]
    
    chain_mask = torch.tensor(
        np.array([ item in chains_to_design_list for item in protein_dict["chain_letters"]], 
        dtype=np.int32), 
        device=device,
    )
    if len(var_residues) > 0:
        fixed_positions = torch.tensor( 
            [int(item.strip() in var_residues) for item in encoded_residues], 
            device=device, 
        )
    else:
        fixed_positions = torch.tensor( 
            [int(item.strip() not in fixed_residues) for item in encoded_residues], 
            device=device, 
        )

    protein_dict['fixed_positions'] = fixed_positions
    protein_dict["chain_mask"] = chain_mask * fixed_positions

    if args.verbose:
        mask = protein_dict["chain_mask"].bool()
        PDB_residues_to_be_redesigned = [encoded_residue_dict_rev[i] for i in mask.nonzero(as_tuple=True)[0].tolist()]
        PDB_residues_to_be_fixed = [encoded_residue_dict_rev[i] for i in (~mask).nonzero(as_tuple=True)[0].tolist()]

        print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
        print("These residues will be fixed: ", PDB_residues_to_be_fixed)

    # Symmetry enforcement, homo_oligomer design
    remapped_symmetry_residues, symmetry_weights = get_symmetry(args, encoded_residue_dict)

    if args.seq_model == 'ligandmpnn':
        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
            use_atom_context=use_atom_context,
            number_of_ligand_atoms=atom_context_num,
            model_type='ligand_mpnn',
        )
    elif args.seq_model == 'proteinmpnn':
        feature_dict = featurize(protein_dict, model_type ='protein_mpnn')

    B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.

    # add additional keys to the feature dictionary
    feature_dict["temperature"] = args.temperature
    feature_dict["bias"] = ((-1e8 * design_params['omit_AA'][None, None, :] 
                             + design_params['bias_AA']).repeat([1, L, 1]) 
                            + bias_AA_per_residue[None] 
                            - 1e8 * omit_AA_per_residue[None])
    feature_dict["symmetry_residues"] = remapped_symmetry_residues
    feature_dict["symmetry_weights"] = symmetry_weights
    feature_dict["randn"] = torch.randn([1, feature_dict["mask"].shape[1]], device=device)
    
    return protein_dict, feature_dict

def run_ligandmpnn(args, protein_dict, feature_dict, model, device='cuda'):
    
    if "Y" in list(protein_dict):
        atom_coords = protein_dict["Y"].cpu().numpy()
        atom_types = list(protein_dict["Y_t"].cpu().numpy())
        atom_mask = list(protein_dict["Y_m"].cpu().numpy())
        number_of_atoms_parsed = np.sum(atom_mask)
    else:
        #print("No ligand atoms parsed")
        number_of_atoms_parsed = 0
        atom_types = ""
        atom_coords = []

    if args.verbose:
        print(f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}")
        if number_of_atoms_parsed > 0:
            for i, atom_type in enumerate(atom_types):
                print(f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}")

    with torch.inference_mode():
        output_dict = model.sample(feature_dict)
    
    # compute confidence scores
    loss, loss_per_residue = get_score(
        output_dict["S"],
        output_dict["log_probs"],
        feature_dict["mask"] * feature_dict["chain_mask"])
    
    #combined_mask = (feature_dict["mask"] * feature_dict["mask_XY"] * feature_dict["chain_mask"])
    #loss_XY, _ = get_score(output_dict["S"], output_dict["log_probs"], combined_mask)

    S = output_dict["S"]
    S_ = torch.transpose(S, 1, 2)
    log_probs = output_dict["log_probs"]
    sampling_probs = output_dict["sampling_probs"]
    decoding_order = output_dict["decoding_order"]
    
    #rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
    #rec = get_seq_rec(feature_dict["S"][:1], S_, rec_mask)

    loss_ = torch.t(loss).detach()
    #loss_XY_ = torch.t(loss).detach()
    #seq_rec_print = np.format_float_positional(
    #    rec.cpu().numpy(), unique=False, precision=4)

    loss_np = np.format_float_positional(
        np.exp(-loss_[0].cpu().numpy()), unique=False, precision=4)
    #loss_XY_np = np.format_float_positional(
    #    np.exp(-loss_XY_[0].cpu().numpy()),
    #    unique=False,
    #    precision=4)

    seq = "".join(
        [restype_int_to_str[AA] for AA in S_.squeeze().cpu().numpy()])

    out_dict = {}
    out_dict["generated_sequence"] = S.detach().cpu()
    out_dict["generated_sequence_str"] = seq
    out_dict["sampling_probs"] = sampling_probs.detach().cpu()
    out_dict["log_probs"] = log_probs.cpu()
    out_dict["decoding_order"] = decoding_order.detach().cpu()
    out_dict["native_sequence"] = feature_dict["S"][0].detach().cpu()
    out_dict["mask"] = feature_dict["mask"][0].detach().cpu()
    out_dict["chain_mask"] = feature_dict["chain_mask"][0].detach().cpu()
    out_dict["seed"] = args.seed
    out_dict["temperature"] = args.temperature
    out_dict['loss_np'] = loss_np

    return out_dict
    
