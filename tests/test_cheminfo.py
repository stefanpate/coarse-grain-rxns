from cgr.cheminfo import (
    rc_neighborhood,
    MorganFingerprinter,
)
from rdkit import Chem

def test_fingerprinter():
    substrate_smiles = 'NC(CC(=O)O)C(=O)O'
    rc = [1, 6, 8]
    substrate_mol = Chem.MolFromSmiles(substrate_smiles)
    rc_neighborhood(substrate_mol, radius=2, reaction_center=rc)

    mfper = MorganFingerprinter(radius=2, length=2**10)
    mfp = mfper.fingerprint(substrate_mol, rc)
    mfp = mfper.fingerprint(substrate_mol, rc, rc_dist_ub=None)

    mfper = MorganFingerprinter(radius=2, length=2**10, allocate_ao=True)
    mfp = mfper.fingerprint(substrate_mol, rc)
    ac = mfper.atom_counts
    bim = mfper.bit_info_map
    a2b = mfper.atom_to_bits
    mfp = mfper.fingerprint(substrate_mol, rc, rc_dist_ub=0)
    ac = mfper.atom_counts
    bim = mfper.bit_info_map
    a2b = mfper.atom_to_bits

# TODO: construct test
# import json
# from cgr.filepaths import filepaths
# from collections import defaultdict
# import pandas as pd

# krs = filepaths.data / "raw" / "sprhea_240310_v3_mapped_no_subunits.json"
# with open(krs, 'r') as f:
#     krs = json.load(f)

# decarb = {k: v for k,v  in krs.items() if v['min_rule'] == 'rule0024'}
# print(len(decarb))

# max_hops = 3
# vec_len = 2**12
# mfper = MorganFingerprinter(radius=max_hops, length=vec_len, allocate_ao=True)
# rc_dist_ub = None
# n_samples = len(decarb)
# data = defaultdict(list)
# for rid, rxn in decarb.items():
#     rc = rxn['reaction_center'][0]
#     smiles = rxn['smarts'].split('>>')[0]
#     mol = Chem.MolFromSmiles(smiles)
#     _ = mfper.fingerprint(mol, reaction_center=rc, rc_dist_ub=rc_dist_ub)
#     bim = mfper.bit_info_map


#     for bit_idx, examples in bim.items():
#         for (central_aidx, radius) in examples:
#             data['feature_id'].append(bit_idx)
#             data['sample_id'].append(rid)

#             sub_idxs, sub_mol, sub_smi = extract_subgraph(mol, central_aidx, radius)

#             data['sub_idxs'].append(sub_idxs)
#             data['sub_smi'].append(sub_smi)
#             data['saturated'].append(is_subgraph_saturated(mol, rc, sub_idxs))
#             data['only_carbon'].append(has_subgraph_only_carbons(mol, rc, sub_idxs))

# raw_subgraphs = pd.DataFrame(data)

# resolved, embeddings = resolve_bit_collisions(raw_subgraphs, vec_len)
# print()

# for elt in resolved['feature_id'].unique():
#     smiles = set(resolved.loc[resolved['feature_id'] == elt, 'sub_smi'].to_list())
#     assert len(smiles) == 1
