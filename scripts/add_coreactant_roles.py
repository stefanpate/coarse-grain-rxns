import hydra
from omegaconf import DictConfig
from itertools import product, combinations
from collections import defaultdict
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def do_clash(this: tuple[tuple[str, str], tuple[str, str]], other: tuple[tuple[str, str], tuple[str, str]]) -> bool:
    return len(set(this[0]).intersection(set(other[0]))) > 0

def make_unclashing_pair_combos(smi_role_pairs: list[tuple[tuple[str, str], tuple[str, str]]]) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    '''
    Form all k combinations of paired role assigments for k=1, ..., len(smi_role_pairs)
    that do not clash, i.e., have the same SMILES in any of the pairs.

    Args
    ----
    smi_role_pairs: list[tuple[tuple[str], tuple[str]]]
        Each element is a tuple of two tuples, where the first tuple contains
        smiles of rct and pdt and the second tuple contains their respective roles.
    Returns
    -------
    list[tuple[tuple[str], tuple[str]]]
        Each element is a combination of paired role assignments, with
        first inner tuple containing smiles and second inner tuple containing roles.
    '''
    max_k = len(smi_role_pairs)

    unclashing_combos = []
    for k in range(1, max_k + 1):
        if k == 1:
            unclashing_combos.extend(smi_role_pairs)
            continue
        
        k_candidates = list(combinations(smi_role_pairs, k))
        k_combos = []
        for candidate in k_candidates:
            clash_found = False
            for i, this in enumerate(candidate):
                for other in candidate[i+1:]:
                    if do_clash(this, other):
                        clash_found = True
                        break
                if clash_found:
                    break
            
            if not clash_found:
                smi_side = []
                role_side = []
                for pair in candidate:
                    smi_side.extend(pair[0])
                    role_side.extend(pair[1])
                
                candidate = (tuple(smi_side), tuple(role_side))
                k_combos.append(candidate)

        unclashing_combos.extend(k_combos)
    
    return unclashing_combos

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

    rcts, pdts = [side.split('.') for side in reaction.split('>>')]
    rct_pdt_pairs = list(product(rcts, pdts))
    smi2idx = defaultdict(list)
    for i, smi in enumerate(rcts + pdts):
        smi2idx[smi].append(i)
    tot = len(rcts) + len(pdts)

    unpaired_roles = defaultdict(set)
    for smi in rcts + pdts:
        unpaired_roles[smi].add(unpaired_smi_to_role.get(smi, default_role))

    paired_roles = []
    for pair in rct_pdt_pairs:
        if pair in paired_smi_to_role:
            paired_roles.append((pair, paired_smi_to_role[pair]))
        
    unclashing_paired_roles = make_unclashing_pair_combos(paired_roles)

    # Cartesian product of roles indexed by unique smiles
    # st role assignments for stoichiometric multiples will
    # be tied
    _roles = product(*unpaired_roles.values())

    # Now unpack the stoich multiples
    roles = []
    for _role in _roles:
        role = [None for _ in range(tot)]
        for smi, single_role in zip(unpaired_roles.keys(), _role):
            for idx in smi2idx[smi]:
                role[idx] = single_role
        roles.append(role)
    
    # Add in all paired role assigment combos
    w_paired_overlaid = []
    for role in roles:
        new_role = [elt for elt in role]
        for paired_role in unclashing_paired_roles:
            for s, r in zip(*paired_role):
                for idx in smi2idx[s]:
                    new_role[idx] = r
        w_paired_overlaid.append(new_role)

    roles.extend(w_paired_overlaid)

    tmp = []
    for role in roles:
        lhs = ";".join(role[:len(rcts)])
        rhs = ";".join(role[len(rcts):])
        tmp.append((lhs, rhs))

    roles = tmp

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
            reaction=row['smarts'],
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