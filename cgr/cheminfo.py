from rdkit import Chem
import re
from itertools import combinations

def get_patts_from_operator_side(smarts: str, side: int) -> list:
    '''
    Get molecule SMARTS patterns from rule SMARTS

    Args
    ----
    smarts:str
        A.B>>C.D
    side:int
        Get from left side if = 0,
        from right side if = 1 
    '''

    # Side smarts pattern
    smarts = smarts.split('>>')[side]
    smarts = re.sub(r':[0-9]+]', ']', smarts)

    # identify each fragment
    fragemented_smarts = []
    tmp = []

    # append complete fragments only
    for fragment in smarts.split('.'):
        tmp += [fragment]
        if '.'.join(tmp).count('(') == '.'.join(tmp).count(')'):
            fragemented_smarts.append('.'.join(tmp))
            tmp = []

            # remove component grouping for substructure matching
            if '.' in fragemented_smarts[-1]:
                fragemented_smarts[-1] = fragemented_smarts[-1].replace('(', '', 1)[::-1].replace(')', '', 1)[::-1]

    return fragemented_smarts

def rc_neighborhood(molecule: Chem.Mol, radius: int, reaction_center: list[int]) -> Chem.Mol:
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

if __name__ == '__main__':
    substrate_smiles = 'NC(CC(=O)O)C(=O)O'
    r = 10
    rc = [1, 6, 8]
    substrate_mol = Chem.MolFromSmiles(substrate_smiles)
    rc_neighborhood(substrate_mol, radius=r, reaction_center=rc)