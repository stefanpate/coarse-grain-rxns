'''
TODO: Scavenge useful stuff into newer modules and delete this

'''

from rdkit import Chem
from itertools import combinations, product, chain
import numpy as np
import pandas as pd
from typing import Iterable

organic_elements = {
    (0, False): '*',
    (1, False): 'H',
    (6, False): 'C',
    (6, True): 'c',
    (7, False): 'N',
    (7, True): 'n',
    (8, False): 'O',
    (8, True): 'o',
    (9, False): 'F',
    (11, False): 'Na',
    (12, False): 'Mg',
    (16, False): 'S',   
    (17, False): 'Cl',  
    (35, False): 'Br',  
    (53, False): 'I',
    (15, False): 'P',
}

def rc_neighborhood(molecule: Chem.Mol, radius: int, reaction_center: Iterable[int]) -> Chem.Mol:
    '''
    Returns subgraph of molecule consisting of all atoms w/in
    bondwise radius of reaction center
    '''
    bidxs = []
    if radius == 0:
        for a, b in combinations(reaction_center, 2):
            bond = molecule.GetBondBetweenAtoms(a, b)
            if bond:
                bidxs.append(bond.GetIdx())

    else:
        for aidx in reaction_center:
            env = Chem.FindAtomEnvironmentOfRadiusN(
                mol=molecule,
                radius=radius,
                rootedAtAtom=aidx
            )
            bidxs += list(env)

        # Beyond full molecule
        if not bidxs:
            bidxs = [bond.GetIdx() for bond in molecule.GetBonds()]


    bidxs = list(set(bidxs))

    submol = Chem.PathToSubmol(
        mol=molecule,
        path=bidxs
    )

    return submol

def extract_subgraph(mol: Chem.Mol, aidx: int, radius: int):
    '''
    Args
    -----
    mol: Chem.Mol
        Molecule
    aidx: int
        Central atom index of the subgraph
    radius: int
        # of hops (bonds) out from the central atom index
    Returns
    -------
    subgraph_aidxs: tuple[int]
        Atom indices of the subgraph
    subgraph_mol: Chem.Mol
        Mol object of subgraph
    subgraph_smiles: str
        SMILES of subgraph
    '''
    subgraph_aidxs = set()

    if radius == 0:
        subgraph_aidxs.add(aidx)
        atom = mol.GetAtomWithIdx(aidx)
        subgraph_smiles = organic_elements[(atom.GetAtomicNum(), atom.GetIsAromatic())]
        subgraph_mol = Chem.MolFromSmiles(subgraph_smiles, sanitize=False)

    else:

        env = Chem.FindAtomEnvironmentOfRadiusN(
            mol=mol,
            radius=radius,
            rootedAtAtom=aidx
        )

        for bidx in env:
            bond = mol.GetBondWithIdx(bidx)
            subgraph_aidxs.add(bond.GetBeginAtomIdx())
            subgraph_aidxs.add(bond.GetEndAtomIdx())

        subgraph_mol = Chem.PathToSubmol(
            mol=mol,
            path=env    
        )
        subgraph_smiles = Chem.MolToSmiles(subgraph_mol)

    return tuple(subgraph_aidxs), subgraph_mol, subgraph_smiles

def is_subgraph_saturated(mol: Chem.Mol, rc: tuple[int], sub_idxs: tuple[int]):
    '''
    Returns True if no bonds in non-rc subgraph greater than single
    '''
    idxs_sans_rc = [i for i in sub_idxs if i not in rc]
    bonds_outside_rc = combinations(idxs_sans_rc, 2)
    bonds_crossing_into_rc = product(idxs_sans_rc, rc)
    for (i, j) in chain(bonds_outside_rc, bonds_crossing_into_rc):
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond and bond.GetBondTypeAsDouble() > 1.0:
            return False
        
    return True

def has_subgraph_only_carbons(mol: Chem.Mol, rc: tuple[int], sub_idxs: tuple[int]):
    '''
    Returns true if only peri-rc element is non-aromatic carbon
    '''
    for idx in sub_idxs:
        atom = mol.GetAtomWithIdx(idx)
        
        if idx in rc:
            if atom.GetIsAromatic():
                return False
        elif atom.GetAtomicNum() != 6 and not atom.GetIsAromatic():
            return False
        
    return True

def subgraph_contains_rc_atoms(rc: tuple[int], sub_idxs: tuple[int]):
    '''
    Returns true if subgraph contains any rc atoms.
    TODO: Delete when you find a more elegant solution
    '''
    for idx in sub_idxs:
        if idx in rc:
            return True
        
    return False

def tanimoto_similarity(bitvec1: np.ndarray, bitvec2: np.ndarray):
    dot = np.dot(bitvec1, bitvec2)
    return dot / (bitvec1.sum() + bitvec2.sum() - dot)

def resolve_bit_collisions(unresolved: pd.DataFrame, n_features: int):
    '''
    Resolved bit collisions occuring over an ensemble of binary
    feature matrices

    Args
    ----
    unresolved:pd.DataFrame
        feature_id | sample_id | sub_smi

    Returns
    -------
    resolved:pd.DataFrame
        Resolved feature df (same cols as input)
    embeddings:pd.DataFrame
        sample_id | embedding:ndarray
    '''

    resolved = []
    unique_ftids = sorted(set(unresolved['feature_id'].to_list()), reverse=False)
    unoccupied = sorted([i for i in range(n_features) if i not in unique_ftids], reverse=True)
    for id in unique_ftids:
        unique_structures = set(unresolved.loc[unresolved['feature_id'] == id, 'sub_smi'])

        if len(unique_structures) == 1:
            resolved.append(unresolved.loc[unresolved['feature_id'] == id, :])
        else:
            unique_structures = sorted(unique_structures)
            for i, us in enumerate(unique_structures):
                sel = (unresolved['feature_id'] == id) & (unresolved['sub_smi'] == us)
                if i == 0:
                    resolved.append(unresolved.loc[sel, :])
                else:
                    to_append = unresolved.loc[sel, :].copy()
                    to_append['feature_id'] = unoccupied.pop()
                    resolved.append(to_append)

    resolved = pd.concat(resolved, ignore_index=True)

    data = []
    unique_samples = sorted(set(resolved['sample_id'].to_list()))
    for usamp in unique_samples:
        embed = np.zeros(shape=(n_features, ))
        arg_nz = resolved.loc[resolved['sample_id'] == usamp, 'feature_id'].to_numpy()
        embed[arg_nz] = 1
        data.append((usamp, embed))

    embeddings = pd.DataFrame(data=data, columns=['sample_id', 'embedding'])

    return resolved, embeddings            

if __name__ == '__main__':
    import json
    import pandas as pd
    from pathlib import Path

    krs = Path("/home/stef/cgr") / "raw" / "sprhea_240310_v3_mapped_no_subunits.json"
    with open(krs, 'r') as f:
        krs = json.load(f)

    decarb = {k: v for k,v  in krs.items() if v['min_rule'] == 'rule0024'}
    print(len(decarb))

    smiles = [v['smarts'].split('>>')[0] for v in decarb.values()]
    rcs = [v['reaction_center'][0] for v in decarb.values()]
    few_smiles = smiles[:3]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    few_mols = [Chem.MolFromSmiles(smi) for smi in few_smiles]
    few_rcs = rcs[:3]
    