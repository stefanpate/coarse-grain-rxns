import hydra
from omegaconf import DictConfig
from ergochemics.standardize import standardize_smiles
from rdkit import Chem
from itertools import product
from collections import defaultdict
from typing import Iterable
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def make_pair_idx_combos(pair_idxs: Iterable[tuple[int]]) -> list[tuple[tuple[int]]]:
    '''
    Form all combinattions of pairs indices that use as many pairs as possible
    without any of them clashing, i.e., having the same index in both pairs.

    Args
    ----
    pair_idxs:Iterable[tuple[int]]
        Each element contains a LHS and RHS molecule index making up a pair
        of molecules that was mapped to a paired coreactant role.

    Returns
    -------
    list[tuple[tuple[int]]]
        Each element is a combination of index pairs, where each pair
        therein has no overlapping indices w/ any other pair in the combination.

    Example
    -------
    pair_idxs = [(0, 1), (1, 2), (2, 3), (4, 5), (8, 9)]
    make_pair_idx_combos(pair_idxs)
    Output: [((0, 1), (8, 9), (4, 5)), ((1, 2), (8, 9), (4, 5)), ((2, 3), (8, 9), (4, 5))]
    '''

    do_clash = lambda x, y: len(set(x).intersection(set(y))) > 0
    S = np.zeros((len(pair_idxs), len(pair_idxs)), dtype=np.int32)
    for i, pi in enumerate(pair_idxs):
        for j, pj in enumerate(pair_idxs):
            if do_clash(pi, pj):
                S[i, j] = 1

    D = 1 - S
    ac = AgglomerativeClustering(
        metric='precomputed',
        linkage='single',
        distance_threshold=0.1,
        n_clusters=None,
    )
    ac.fit(D)

    labels = ac.labels_
    clusters = defaultdict(list)
    for lab in np.unique(labels):
        clusters[lab] = [pair_idxs[elt] for elt in np.where(labels == lab)[0]]

    return list(product(*clusters.values()))

def get_coreactant_roles(reaction: str, unpaired_smi_to_role: dict[str, str], paired_smi_to_role: dict[tuple[str], tuple[str]]) -> list[tuple[str, str]]:
    """
    Get the coreactant roles for a given reaction.

    Args
    ----
    reaction: str
        The reaction string.
    unpaired_smi_to_role: dict[str, str]
        A dictionary mapping unpaired SMILES to their roles.
    paired_smi_to_role: dict[tuple[str], tuple[str]]
        A dictionary mapping paired SMILES to their roles.

    Returns
    -------
    roles:list[tuple[str, str]]
        Each element has a tuple of strings, one for each side of the reaction.
        The tuple elements give roles for each molecule on a side of the reaction,
        separated by a semicolon.

    Notes
    -----
    - The default role is "Any"
    """
    default_role = "Any"

    rcts, pdts = [[Chem.MolFromSmiles(mol) for mol in elt.split('.')] for elt in reaction.split('>>')]
    
    # Ensure SMILES de-atom-mapped and standardized for lookups in smi_to_role dicts
    for mol in rcts + pdts:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    rcts = [standardize_smiles(Chem.MolToSmiles(mol)) for mol in rcts]
    pdts = [standardize_smiles(Chem.MolToSmiles(mol)) for mol in pdts]

    up_lhs_roles = {}
    up_rhs_roles = {}

    # Do unpaired first
    for i, rct in enumerate(rcts):
        up_lhs_roles[i] = unpaired_smi_to_role.get(rct, default_role)

    for i, pdt in enumerate(pdts):
        up_rhs_roles[i] = unpaired_smi_to_role.get(pdt, default_role)

    # Do paired second, in some cases, pairs will overwrite unpaired
    pair_idx_candidates = []
    for idx_pair in product(up_lhs_roles.keys(), up_rhs_roles.keys()):
        pair = (rcts[idx_pair[0]], pdts[idx_pair[1]])
        if pair in paired_smi_to_role:
            pair_idx_candidates.append(idx_pair)

    if len(pair_idx_candidates) == 0:
        pair_idx_combos = []
    elif len(pair_idx_candidates) == 1:
        pair_idx_combos = [tuple(pair_idx_candidates)]
    else: # Get all unclashing combinations of paired role assignments
        pair_idx_combos = make_pair_idx_combos(pair_idx_candidates)

    roles = []
    if len(pair_idx_combos) == 0:
        lhs_roles = {k: v for k, v in up_lhs_roles.items()}
        rhs_roles = {k: v for k, v in up_rhs_roles.items()}
        roles.append((";".join(lhs_roles.values()), ";".join(rhs_roles.values())))
    else:
        for pair_idx_combo in pair_idx_combos:
            lhs_roles = {k: v for k, v in up_lhs_roles.items()}
            rhs_roles = {k: v for k, v in up_rhs_roles.items()}
            for idx_pair in pair_idx_combo:
                pair = (rcts[idx_pair[0]], pdts[idx_pair[1]])
                paired_role = paired_smi_to_role[pair]
                lhs_roles[idx_pair[0]] = paired_role[0]
                rhs_roles[idx_pair[1]] = paired_role[1]

            roles.append((";".join(lhs_roles.values()), ";".join(rhs_roles.values())))

    return roles

@hydra.main(version_base=None, config_path='../configs', config_name='add_coreactant_roles')
def main(cfg: DictConfig):
    # Load data
    mapped_rxns = pd.read_parquet(Path(cfg.filepaths.raw_data) / cfg.mapped_rxns_fn)

    # Load coreactant lookups
    unpaired_smi_to_role = {}
    paired_smi_to_role = {}
    up_ref = pd.read_csv(Path(cfg.filepaths.coreactants) / "unpaired.tsv", sep='\t')
    p_ref = pd.read_csv(Path(cfg.filepaths.coreactants) / "paired.tsv", sep='\t')
    for _, row in up_ref.iterrows():
        unpaired_smi_to_role[row['smiles']] = row['class']

    for _, row in p_ref.iterrows():
        paired_smi_to_role[(row['smiles_1'], row['smiles_2'])] = (row['class_1'], row['class_2'])
        paired_smi_to_role[(row['smiles_2'], row['smiles_1'])] = (row['class_2'], row['class_1']) # Reverse all paired roles

    # Get coreactant roles
    rule_2_role_templates = defaultdict(set)
    for _, row in tqdm(mapped_rxns.iterrows(), total=len(mapped_rxns)):
        roles = get_coreactant_roles(
            reaction=row['am_smarts'],
            unpaired_smi_to_role=unpaired_smi_to_role,
            paired_smi_to_role=paired_smi_to_role,    
        )
        for role in roles:
            rule_2_role_templates[row['rule_id']].add(role)

    # Save to .tsv w/ columns as Pickaxe expects
    data = {
        'Name': [],
        'Reactants': [],
        'SMARTS': [],
        'Products': [],
    }
    for base_rule_id, role_templates in rule_2_role_templates.items():
        for i, role_template in enumerate(sorted(role_templates)):
            data['Name'].append(f"{base_rule_id}_{i}")
            data['Reactants'].append(role_template[0])
            data['SMARTS'].append(mapped_rxns.loc[mapped_rxns['rule_id'] == base_rule_id].iloc[0]['rule'])
            data['Products'].append(role_template[1])

    rules_w_coreactants = pd.DataFrame(data)
    rules_w_coreactants.to_csv(f"{Path(cfg.mapped_rxns_fn).stem.split('_x_')[-1]}_w_coreactants.tsv", sep='\t', index=False)

if __name__ == '__main__':
    main()   