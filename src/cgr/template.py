from rdkit import Chem
from typing import Iterable
import networkx as nx
from itertools import combinations, product
import re

def extract_reaction_template(rxn: str, atoms_to_include: Iterable[Iterable[int]]) -> str:
    '''
    Extracts a reaction template from a reaction string and a list of atoms to include.
    
    Args
    ----
    rxn : str
        Atom-mapped reaction string
    atoms_to_include : Iterable[Iterable[int]]
        List of lists of atom indices to include in the template

    Returns
    -------
    str
        Reaction template (SMARTS)    
    '''
    lhs, rhs = [[Chem.MolFromSmiles(s) for s in side.split('.')] for side in rxn.split('>>')]
    rhs = [[rmol, set(), set()] for rmol in rhs] # Initialize rhs

    ltemplate = []
    for lmol, l_incl_aidxs in zip(lhs, atoms_to_include):
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

        # Collect rhs aidxs corresponding to lhs aidxs
        for i, elt in enumerate(rhs):
            rmol = elt[0]
            elt[1] = elt[1].union(set([atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomMapNum() in incl_amns]))
            elt[2] = elt[2].union(set([atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomMapNum() in connecting_amns]))

        # Construct lhs template
        lsma = get_mol_template(lmol, l_incl_aidxs, l_connecting_aidxs)
        ltemplate.append(lsma)

    # Construct rhs template
    rtemplate = []
    for rmol, r_incl_aidxs, r_connecting_aidxs in rhs:
        rsma = get_mol_template(rmol, r_incl_aidxs, r_connecting_aidxs)
        rtemplate.append(rsma)
    
    canonical_template = canonicalize_template(ltemplate, rtemplate)

    return canonical_template

def canonicalize_template(ltemplate: list[str], rtemplate :list[str]) -> str:
    '''
    Canonicalizes the atom map numbers in the reaction template. Encapsulates disjoint molecules in parentheses.

    Args
    ----
    ltemplate: list[str]
        SMARTS strings for each reactant
    rtemplate: list[str]
        SMARTS strings for each product

    Returns
    -------
    : str
        Canonicalized reaction template
    
    '''
    lmols = [Chem.MolFromSmarts(s) for s in ltemplate]
    rmols = [Chem.MolFromSmarts(s) for s in rtemplate]

    # Remove atom map numbers
    for mol in lmols + rmols:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    
    de_am_ltemplate = [Chem.MolToSmarts(mol) for mol in lmols]
    de_am_rtemplate = [Chem.MolToSmarts(mol) for mol in rmols]

    # Sort by de-atom-mapped SMARTS
    lsrt = sorted(range(len(ltemplate)), key=lambda x: de_am_ltemplate[x])
    rsrt = sorted(range(len(rtemplate)), key=lambda x: de_am_rtemplate[x])
    ltemplate = [ltemplate[i] for i in lsrt]
    rtemplate = [rtemplate[i] for i in rsrt]
    de_am_ltemplate = [de_am_ltemplate[i] for i in lsrt]
    de_am_rtemplate = [de_am_rtemplate[i] for i in rsrt]

    # Sort disjoint intramolecular subgraphs canonically
    for i, elt in enumerate(ltemplate):
        if '.' in elt:
            elt = elt.split('.')
            intra_srt = sorted(range(len(elt)), key=lambda x: de_am_ltemplate[i].split('.')[x])
            ltemplate[i] = ".".join([elt[j] for j in intra_srt])
    
    for i, elt in enumerate(rtemplate):
        if '.' in elt:
            elt = elt.split('.')
            intra_srt = sorted(range(len(elt)), key=lambda x: de_am_rtemplate[i].split('.')[x])
            rtemplate[i] = ".".join([elt[j] for j in intra_srt])
    
    # Reassign atom map numbers in canonical order
    lmols = [Chem.MolFromSmarts(s) for s in ltemplate]
    rmols = [Chem.MolFromSmarts(s) for s in rtemplate]
    rhs_am_to_midx_aidx = {
        atom.GetAtomMapNum(): (i, atom.GetIdx())
        for i, mol in enumerate(rmols)
        for atom in mol.GetAtoms()
    }
    new_am = 1
    for lmol in lmols:
        for atom in lmol.GetAtoms():
            old_am = atom.GetAtomMapNum()
            midx, aidx = rhs_am_to_midx_aidx[old_am]
            rmols[midx].GetAtomWithIdx(aidx).SetAtomMapNum(new_am)
            atom.SetAtomMapNum(new_am)
            new_am += 1

    canonicalized_l_template = []
    for lmol in lmols:
        sma = Chem.MolToSmarts(lmol)
        if '.' in sma:
            sma = '(' + sma + ')' # Encapsulate disjoint molecules
        canonicalized_l_template.append(sma)

    canonicalized_r_template = []
    for rmol in rmols:
        sma = Chem.MolToSmarts(rmol)
        if '.' in sma:
            sma = '(' + sma + ')'
        canonicalized_r_template.append(sma)

    canonicalized_template = ".".join(canonicalized_l_template) + ">>" + ".".join(canonicalized_r_template)
    
    return canonicalized_template

def connect_ccs(G: nx.Graph, ccs: list[set[int]]) -> set[int]:
    '''
    Returns indices of the atoms required to connect disjoint components
    along the shortest paths possible

    Args
    ----
    G: nx.Graph
        Graph representation of the molecule
    ccs: list[set[int]]
        List of connected components

    Returns
    -------
    : set[int]
        Indices of atoms required to connect disjoint components
    '''
    paths = dict(nx.shortest_path(G))
    connecting_aidxs = []
    for cc_i, cc_j in combinations(ccs, 2):
        ij_paths = [paths[i][j] for i, j in product(cc_i, cc_j)]
        min_ij_path = ij_paths[
            min(range(len(ij_paths)), key=lambda x: len(ij_paths[x]))
        ]
        connecting_aidxs.extend(min_ij_path)

    return set(connecting_aidxs)

def get_mol_template(mol: Chem.Mol, incl_aidxs: set[int], cxn_aidxs: set[int]) -> str:
    '''
    Returns SMARTS pattern for a molecule given the indices of atoms to include with identity
    and the indices of atoms to include as anonymous / wildcard.

    Args
    ----
    mol: Chem.Mol
        Molecule
    incl_aidxs: set[int]
        Indices of atoms to include with identity
    cxn_idxs: set[int]
        Indices of atoms to include as wildcard. Through these,
        the named atoms from disjoint components are connected by the
        fewest number of hops

    Returns
    -------
    : str
        SMARTS pattern for the molecule
    '''
    amn_to_aidx = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() > 0: # Atom mapping provided
            amn_to_aidx[atom.GetAtomMapNum()] = atom.GetIdx()
        else: # No atom mapping provided
            amn_to_aidx[atom.GetIdx() + 1] = atom.GetIdx()

    for aidxs in cxn_aidxs:
        mol.GetAtomWithIdx(aidxs).SetAtomicNum(0) # Mark wildcard atoms

    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in cxn_aidxs or bond.GetEndAtomIdx() in cxn_aidxs:
            bond.SetBondType(Chem.rdchem.BondType.ZERO) # Mark wildcard bonds
    
    sma = Chem.MolFragmentToSmarts(mol, atomsToUse=incl_aidxs | cxn_aidxs)
    sma = re.sub(r'\[#0H?\d?\+?', '[*', sma) # Wildcard atoms
    sma = re.split(r'(\[#\d{1,3}H?\d?\+?:\d{1,3}\])', sma) # Split on atom patterns
    
    tmp = []
    for i, elt in enumerate(sma):
        if elt == '':
            continue
        
        if i % 2 == 1: # Atomic patts
            amn = int(elt.strip('[]').split(':')[-1]) # Hard brackets will return from atomic patt replacements
            repl = get_atom_smarts(mol.GetAtomWithIdx(amn_to_aidx[amn]))
            tmp.append(repl) # Replace with the SMARTS for the atom
        else: # Bond patts
            tmp.append(elt) # Hard brackets will return from atomic patt replacements

    sma = "".join(tmp) # Join back together

    return sma

def get_atom_smarts(
    atom: Chem.Atom,
    degree: bool = True,
    valence: bool = True,
    hydgrogens: bool = True,
    aromatic: bool = True,
    formal_charge: bool = True,
    in_ring: bool = True,
    heteroneighbors: bool = True,
    atom_map_num: bool = True
    ) -> str:
    symbol = atom.GetSymbol()

    if aromatic and atom.GetIsAromatic():
        symbol = symbol.lower()
    
    qualifiers = []
    qualifiers.append(symbol)

    if degree:
        qualifiers.append(f"D{atom.GetDegree()}")
    
    if valence:
        qualifiers.append(f"v{atom.GetTotalValence()}")

    if hydgrogens:
        qualifiers.append(f"H{atom.GetTotalNumHs()}")

    if formal_charge:
        formal_charge_value = atom.GetFormalCharge()
        if formal_charge_value > 0:
            qualifiers.append(f"+{abs(formal_charge_value)}")
        elif formal_charge_value < 0:
            qualifiers.append(f"-{abs(formal_charge_value)}")
        else:
            qualifiers.append("0")

    if in_ring:
        if atom.IsInRing():
            qualifiers.append("R")
        else:
            qualifiers.append("!R")
    
    if heteroneighbors and atom.GetAtomicNum() == 6: # Only apply to carbon
        het_ct = 0
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 6:
                het_ct += 1
        qualifiers.append(f"z{het_ct}")

    atomic_patt = "&".join(qualifiers)

    if atom_map_num:
        atomic_patt += f":{atom.GetAtomMapNum()}"

    return f"[{atomic_patt}]"

if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd
    from itertools import chain

    to_nested_lists = lambda x: [[arr.tolist() for arr in side] for side in x]

    mm = pd.read_parquet(
    Path("/home/stef/cgr/data/raw") / "mapped_mech_labeled_reactions.parquet"
    )
    # test = mm.loc[mm['entry_id'] == 49]
    for _, test in mm.iterrows():
        rc = to_nested_lists(test['reaction_center'])
        mech_atoms = to_nested_lists(test['mech_atoms'])
        atoms_to_include = [chain(*elt) for elt in zip(rc[0], mech_atoms)]
        am_smarts = test['am_smarts']
        template = extract_reaction_template(am_smarts, atoms_to_include)
        print(template)
    
    print()
