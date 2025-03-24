from rdkit import Chem
from typing import Iterable
import networkx as nx
from itertools import combinations, product
import re

def extract_rule_template(rxn: str, atoms_to_include: Iterable[Iterable[int]]) -> str:
    lhs, rhs = [[Chem.MolFromSmiles(s) for s in side.split('>>')] for side in rxn.split('>>')]

    ltemplate, rtemplate = [], []
    for lmol, rmol, l_incl_aidxs in zip(lhs, rhs, atoms_to_include):
        l_incl_aidxs = set(l_incl_aidxs)
        A = Chem.GetAdjacencyMatrix(lmol)
        A_incl = A[tuple(l_incl_aidxs), :][:, tuple(l_incl_aidxs)]
        G_incl = nx.from_numpy_array(A_incl)
        ccs = list(nx.connected_components(G_incl))

        if len(ccs) > 1:
            G = nx.from_numpy_array(A)
            l_connecting_aidxs = connect_ccs(G, ccs)
            l_connecting_aidxs = l_connecting_aidxs - l_incl_aidxs
            connecting_amns = [lmol.GetAtomWithIdx(idx).GetAtomMapNum() for idx in l_connecting_aidxs]
        else:
            l_connecting_aidxs = set()
            connecting_amns = []
        
        incl_amns = [lmol.GetAtomWithIdx(idx).GetAtomMapNum() for idx in l_incl_aidxs]
        r_incl_aidxs = set([atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomMapNum() in incl_amns])
        r_connecting_aidxs = set([atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomMapNum() in connecting_amns])


        lsma = get_mol_template(lmol, l_incl_aidxs, l_connecting_aidxs)
        rsma = get_mol_template(rmol, r_incl_aidxs, r_connecting_aidxs)
        ltemplate.append(lsma)
        rtemplate.append(rsma)

    template = ".".join(ltemplate) + ">>" + ".".join(rtemplate)


    canonical_template = canonicalize_template(template)

    return canonical_template


def canonicalize_template(template: str) -> str:
    lsma, rsma = [side.split(".") for side in template.split(">>")]
    lmols = [Chem.MolFromSmarts(s) for s in lsma]
    rmols = [Chem.MolFromSmarts(s) for s in rsma]

    # Remove atom map numbers
    for mol in lmols + rmols:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    
    de_am_lsma = [Chem.MolToSmarts(mol) for mol in lmols]
    de_am_rsma = [Chem.MolToSmarts(mol) for mol in rmols]
    lsrt = sorted(range(len(lsma)), key=lambda x: de_am_lsma[x]) # Sort by de-atom-mapped SMARTS
    rsrt = sorted(range(len(rsma)), key=lambda x: de_am_rsma[x])
    lsma = ".".join([lsma[i] for i in lsrt])
    rsma = ".".join([rsma[i] for i in rsrt])
    lmol = Chem.MolFromSmarts(lsma)
    rmol = Chem.MolFromSmarts(rsma)
    
    # Reassign atom map numbers in canonical order
    rhs_am_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rmol.GetAtoms()}
    new_am = 1
    for atom in lmol.GetAtoms():
        old_am = atom.GetAtomMapNum()
        rmol.GetAtomWithIdx(rhs_am_to_idx[old_am]).SetAtomMapNum(new_am)
        atom.SetAtomMapNum(new_am)
        new_am += 1

    canonicalized_template = Chem.MolToSmarts(lmol) + ">>" + Chem.MolToSmarts(rmol)
    
    return canonicalized_template

def get_mol_template(mol: Chem.Mol, incl_aidxs: set[int], cxn_idxs: set[int]) -> str:
    aidx_to_amn = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()} # Save aidx2amn
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0) # Remove amns from mol

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) # Remove H information

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(aidx_to_amn[atom.GetIdx()]) # Re-establish amns
    
    for aidxs in cxn_idxs:
        mol.GetAtomWithIdx(aidxs).SetAtomicNum(0) # Mark wildcard atoms

    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in cxn_idxs or bond.GetEndAtomIdx() in cxn_idxs:
            bond.SetBondType(Chem.rdchem.BondType.ZERO) # Mark wildcard bonds
    
    sma = Chem.MolFragmentToSmarts(mol, atomsToUse=incl_aidxs | cxn_idxs)
    sma = re.sub(r'\[#0H*\d*', '[*', sma) # Wildcard atoms

    return sma

def connect_ccs(G: nx.Graph, ccs: list[set[int]]) -> set[int]:
    paths = dict(nx.shortest_path(G))
    connecting_aidxs = []
    for cc_i, cc_j in combinations(ccs, 2):
        ij_paths = [paths[i][j] for i, j in product(cc_i, cc_j)]
        min_ij_path = ij_paths[
            min(range(len(ij_paths)), key=lambda x: len(ij_paths[x]))
        ]
        connecting_aidxs.extend(min_ij_path)

    return set(connecting_aidxs)

if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd
    from itertools import chain

    to_nested_lists = lambda x: [[arr.tolist() for arr in side] for side in x]

    mm = pd.read_parquet(
    Path("/home/stef/cgr/data/raw") / "mapped_mech_labeled_reactions.parquet"
    )
    test = mm.loc[mm['entry_id'] == 49]
    rc = to_nested_lists(test['reaction_center'].iloc[0])
    mech_atoms = to_nested_lists(test['mech_atoms'].iloc[0])
    atoms_to_include = [chain(*elt) for elt in zip(rc[0], mech_atoms)]
    am_smarts = test['am_smarts'].iloc[0]
    template = extract_rule_template(am_smarts, atoms_to_include)
    print(template)
    
    print()
