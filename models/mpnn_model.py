import gc
import os
import sys
import torch
import logging
from typing import Dict

from common.utils import ensure_dir, create_file
from core.interfaces import BaseModel
from core.registry import register_model
from evolve.individual import Individual

from .ligandmpnn.ligandmpnn import *
from .ligandmpnn.pdb_utils import *
from .ligandmpnn.sc_utils import Packer
from .ligandmpnn.model_utils import ProteinMPNN
from .ligandmpnn.data_utils import restype_int_to_str, parse_PDB

logger = logging.getLogger("evolution")
logger.setLevel(logging.DEBUG)

@register_model("seq_model")
class MPNNModel(BaseModel):

    def __init__(self):
        pass
    
    def setup(self, config: Dict, device: str='cpu') -> None:
        self.config = config
        self.device = device
        self.seed = self.config.general.seed
        self.seq_model = self.config.seq_model

        if self.seq_model == 'ligandmpnn':
            self.model_config = self.config.models.ligandmpnn
            self.model_config.seq_model = self.seq_model
            self.model_config.seed = self.seed
        elif self.seq_model == 'proteinmpnn':
            self.model_config = self.config.models.proteinmpnn
            self.model_config.seq_model = self.seq_model
            self.model_config.seed = self.seed
        else:
            assert False, 'Sequence model not properly configured.'
        
        outputs = self.config.general.outputs
        self.output_fasta = os.path.join(outputs, "seqs")
        self.output_packed = os.path.join(outputs, "packed")
        ensure_dir(self.output_fasta)
        ensure_dir(self.output_packed)

        self.pdb = self.config.input.pdb
        self.pdb_name = os.path.split(self.pdb)[-1].split('.')[0]
        self.fasta_file = os.path.join(self.output_fasta, f'{self.pdb_name}_s{self.seed}.fasta')

        logger.info(f'Setting up sequence design model {self.seq_model}...')
        self.import_models()
        self.design_constraints()

        # prepare pdb
        self.protein_dict, self.backbone, self.other_atoms, self.icodes, _ = parse_PDB(
            self.pdb, device=self.device, chains=self.parse_these_chains_only_list, parse_all_atoms=True,
            parse_atoms_with_zero_occupancy=self.parse_atoms_with_zero_occupancy)

        # prepare inputs for ligandmpnn - wrapper
        self.protein_dict, self.feature_dict = prepare_ligandmpnn(self.model_config, self.protein_dict, self.icodes,
                                                                self.design_params, self.atom_context_num, device=self.device)
        encoded_residues, encoded_residue_dict, _ = get_encoded_residues(self.protein_dict, self.icodes)
        self.omit_AA_per_residue_tensor = omit_aa(self.model_config, encoded_residues, encoded_residue_dict, 
                                                 self.design_params['omit_AA_per_residue'], self.device)

    def init_seq(self, individual: Individual):
        self.feature_dict["randn"] = torch.randn([1, self.feature_dict["mask"].shape[1]],device=self.device)
        
        # run ligandmpnn
        output_dict = run_ligandmpnn(self.model_config, self.protein_dict,  self.feature_dict, 
            self.model, device=self.device)

        seq_tensor = torch.transpose(output_dict["generated_sequence"], 1,2 )[0]
        seq_str = output_dict["generated_sequence_str"]

        outfile = create_file(self.output_packed, self.pdb_name, individual.get_gen(), individual.get_index(), self.seed)

        # pack sidechains
        _ = pack_sc(self.model_config, self.packer, self.protein_dict, 
                output_dict["generated_sequence"], self.other_atoms, self.icodes,
                outfile=outfile, device=self.device)

        torch.cuda.empty_cache()
        gc.collect()
        
        individual.update_name(name=outfile)
        individual.update_seq_str(seq_str=seq_str)
        individual.update_seq_tensor(seq_tensor=seq_tensor)
        individual.add_fitness({'pmpnn':float(output_dict['loss_np'])})

    def score(self, individual: Individual, num_mutations: int=1):
        protein_dict, _, other_atoms, icodes, _ = parse_PDB(
                individual.name,
                device=self.device,
                chains=self.parse_these_chains_only_list,
                parse_all_atoms=True,
                parse_atoms_with_zero_occupancy=0)

        protein_dict, feature_dict = prepare_ligandmpnn(self.model_config, protein_dict, icodes,
                                        self.design_params, self.atom_context_num, self.device)
        feature_dict["mask"] = self.feature_dict['mask']
        feature_dict['S'] = individual.sequence_.to(self.device)
        score_dict = self.model.single_aa_score(feature_dict, use_sequence=1)
        
        probs_1d, probs_2d = self.get_probs(score_dict, protein_dict, individual.sequence_)
        p = probs_1d/probs_1d.sum() 
        
        L = individual.sequence_.squeeze().shape[0]
        empty_list = np.arange(L)
        
        random_resis = np.random.choice(empty_list, size = num_mutations, replace = False, p = p)
        
        # choose mutation at random based on 2d prob
        try:
            new_resis = utils.sample_indices(probs_2d[random_resis])
        except Exception as e:
            new_resis = torch.argmax(probs_2d[random_resis], dim=1)

        S_temp = score_dict['S'][0].to(self.device)
        S_temp[random_resis] = new_resis.to(self.device)
        seq_tensor = S_temp.unsqueeze(0)
        seq_str = "".join([restype_int_to_str[AA] for AA in S_temp.squeeze().cpu().numpy()])

        score_dict_new = self.model.score(feature_dict, use_sequence=1)
        overall_probs = self.get_probs(score_dict_new, protein_dict, seq_tensor, overall=True)

        outfile = create_file(self.output_packed, self.pdb_name, individual.get_gen(), individual.get_index(), self.seed)

        _ = pack_sc(self.model_config, self.packer, protein_dict, S_temp, other_atoms, 
                        icodes, outfile=outfile, device=self.device)

        individual.update_name(name=outfile)
        individual.update_seq_str(seq_str=seq_str)
        individual.update_seq_tensor(seq_tensor=seq_tensor)
        individual.add_fitness({'pmpnn': np.mean(overall_probs)})

        torch.cuda.empty_cache()
        gc.collect()

    def rescore(self, individual: Individual):
        """Score the individual's current sequence on its current PDB (no mutate/pack).

        Overwrites fitness['pmpnn']. Used after relax so pmpnn reflects
        sequence fit to the post-relax backbone that EF/pocket/desolv also see.
        """
        protein_dict, _, _, icodes, _ = parse_PDB(
            individual.name,
            device=self.device,
            chains=self.parse_these_chains_only_list,
            parse_all_atoms=True,
            parse_atoms_with_zero_occupancy=0)

        protein_dict, feature_dict = prepare_ligandmpnn(
            self.model_config, protein_dict, icodes,
            self.design_params, self.atom_context_num, self.device)
        feature_dict["mask"] = self.feature_dict['mask']
        feature_dict['S'] = individual.sequence_.to(self.device)

        score_dict = self.model.score(feature_dict, use_sequence=1)
        overall_probs = self.get_probs(
            score_dict, protein_dict, individual.sequence_, overall=True)
        individual.update_fitness({'pmpnn': float(np.mean(overall_probs))})

        torch.cuda.empty_cache()
        gc.collect()

    def get_probs(self, score_dict, protein_dict, sequence, overall=False):
        probs_2d = torch.exp(score_dict["log_probs"]).mean(dim=0)
        temp_seq = sequence.squeeze()
        L = temp_seq.shape[0]
        
        probs_1d = probs_2d[torch.arange(L), temp_seq].detach().cpu().numpy().astype('float64')
        if overall:
            return probs_1d
        chain_mask = protein_dict['chain_mask'].detach().cpu().numpy()
        probs_1d = probs_1d * chain_mask
        
        omit_mask = self.omit_AA_per_residue_tensor.detach().cpu().numpy()
        temp_seq_np = temp_seq.detach().cpu().numpy()        
        omit_values = omit_mask[np.arange(L), temp_seq_np]
        omit_positions = (chain_mask == 1) & (omit_values == 1.0)
        
        probs_1d[omit_positions] = 0.0
        
        return probs_1d, probs_2d.detach().cpu()

    def fixed_resis(self, indices=False): 
        return ~self.protein_dict["fixed_positions"].to(self.device).to(torch.bool)

    def design_constraints(self):
        # design constraints
        self.fixed_residues = get_fixed_residues(self.model_config, self.pdb)
        self.var_residues = get_var_residues(self.model_config, self.pdb)

        if len(self.var_residues) > 0 and len(self.fixed_residues) > 0:
            raise ValueError('Can only specify var residues or fixed residues, not both')
        self.bias_AA, self.bias_AA_per_residue = get_bias_aa(self.model_config, self.pdb, self.device)
        self.omit_AA, self.omit_AA_per_residue = get_omit_aa(self.model_config, self.pdb, self.device)
        self.parse_these_chains_only_list = get_parse_chains(self.model_config, self.pdb)
        self.parse_atoms_with_zero_occupancy =self.model_config.parse_atoms_with_zero_occupancy
            
        # compile
        self.design_params = {
            'var_residues': self.var_residues,
            'fixed_residues': self.fixed_residues,
            'bias_AA': self.bias_AA, 
            'bias_AA_per_residue': self.bias_AA_per_residue, 
            'omit_AA': self.omit_AA,
            'omit_AA_per_residue': self.omit_AA_per_residue,
            'parse_these_chains_only_list': self.parse_these_chains_only_list
        }

    def import_models(self):
        if self.seq_model == 'ligandmpnn':
            checkpoint = torch.load(self.model_config.model_path, map_location=self.device)
            k_neighbors = checkpoint["num_edges"]
            self.atom_context_num = checkpoint["atom_context_num"]
            self.model = ProteinMPNN(
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                k_neighbors=k_neighbors,
                device=self.device,
                atom_context_num=self.atom_context_num,
                model_type="ligand_mpnn",
                ligand_mpnn_use_side_chain_context=True,
            )
            self.packer = Packer(
                node_features=128,
                edge_features=128,
                num_positional_embeddings=16,
                num_chain_embeddings=16,
                num_rbf=16,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                atom_context_num=16,
                lower_bound=0.0,
                upper_bound=20.0,
                top_k=32,
                dropout=0.0,
                augment_eps=0.0,
                atom37_order=False,
                device=self.device,
                num_mix=3,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.model.to(self.device)

            checkpoint_packer = torch.load(self.model_config.packer_path, map_location=self.device)
            self.packer.load_state_dict(checkpoint_packer["model_state_dict"])
            self.packer.eval()
            self.packer.to(self.device)

        elif self.seq_model == 'proteinmpnn':
            print('Using ProteinMPNN...')
            checkpoint = torch.load(self.model_config.model_path, map_location=self.device)
            k_neighbors = checkpoint["num_edges"]
            self.atom_context_num = -1
            self.model = ProteinMPNN(
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                k_neighbors=k_neighbors,
                device=self.device,
                model_type="protein_mpnn",
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.model.to(self.device)

            self.model = None
        else:
            assert False, 'Invalid config or model choice...'
