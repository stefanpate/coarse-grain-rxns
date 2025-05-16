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

def make_pair_idx_combos(pair_idxs: Iterable[tuple[int]]) -> list[tuple[int]]:
    '''
    Form all combinattions of pairs indices that use as many pairs as possible
    without any of them clashing, i.e., having the same index in both pairs.
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
    list[tuple[str, str]]
        A list roles, each element has a tuple for each side of the reaction.
        Tuples contain a role name from the provided dicts or the default role
        'Any'
    """
    default_role = "Any"

    rcts, pdts = [[Chem.MolFromSmiles(mol) for mol in elt.spilt('.')] for elt in reaction.split('>>')]
    for mol in rcts + pdts:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    rcts = [standardize_smiles(Chem.MolToSmiles(mol)) for mol in rcts]
    pdts = [standardize_smiles(Chem.MolToSmiles(mol)) for mol in pdts]

    up_lhs_roles = []
    up_rhs_roles = []

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

    # Get all unclashing combinations of pairs
    pair_idx_combos = make_pair_idx_combos(pair_idx_candidates)

    roles = []
    for pair_idx_combo in pair_idx_combos:
        lhs_roles = {k: v for k, v in up_lhs_roles.items()}
        rhs_roles = {k: v for k, v in up_rhs_roles.items()}
        for pair_idx in pair_idx_combo:
            lhs_roles[pair_idx[0]] = paired_smi_to_role[pair_idx]
            rhs_roles[pair_idx[1]] = paired_smi_to_role[pair_idx]

        roles.append((";".join(lhs_roles.values()), ";".join(rhs_roles.values())))

    return roles

@hydra.main(version_base=None, config_path='../configs', config_name='add_coreactant_roles')
def main(cfg: DictConfig):
    pass
    # Load data
    mapped_rxns = pd.read_parquet(Path(cfg.filepaths.raw_data) / cfg.mapped_rxns_fn)
    rules = pd.read_csv(Path(cfg.filepaths.rules) / cfg.rules_fn, sep=',')

    # Load coreactant lookups
    unpaired_smi_to_role = {}
    paired_smi_to_role = {}
    up_ref = pd.read_csv(Path(cfg.filepaths.coreactants) / "unpaired.tsv", sep='\t')
    p_ref = pd.read_csv(Path(cfg.filepaths.coreactants) / "paired.tsv", sep='\t')
    for _, row in up_ref.iterrows():
        unpaired_smi_to_role[row['smiles']] = row['class']

    for _, row in p_ref.iterrows():
        paired_smi_to_role[(row['smiles1'], row['smiles2'])] = (row['class1'], row['class2'])
        paired_smi_to_role[(row['smiles2'], row['smiles1'])] = (row['class2'], row['class1']) # Reverse all paired roles

    # Get coreactant roles

    # For all templates, add rules w/ purely 'Any' roles


    # Save to file
    rules_w_coreactants.to_csv(f"{Path(cfg.rules_fn).stem}_w_coreactants.tsv", sep='\t', index=False)

if __name__ == '__main__':
    # main()

    pair_idxs = [(0, 1), (1, 2), (2, 3), (4, 5), (8, 9)]
    pair_combos = make_pair_idx_combos(pair_idxs)
    print(pair_combos)