from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdFingerprintGenerator
import re
from itertools import combinations
import numpy as np
from typing import Iterable
from functools import partial

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
    organic_elements = {
        0: '*',
        1: 'H',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F',
        11: 'Na',
        12: 'Mg',
        16: 'S',   
        17: 'Cl',  
        35: 'Br',  
        53: 'I',
        15: 'P',
    }

    subgraph_aidxs = set()

    if radius == 0:
        subgraph_aidxs.add(aidx)
        subgraph_smiles = organic_elements[mol.GetAtomWithIdx(aidx).GetAtomicNum()]
        subgraph_mol = Chem.MolFromSmiles(subgraph_smiles)

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

class MorganFingerprinter:
    def __init__(self, radius: int, length: int, allocate_ao: bool = False, **kwargs):
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=length, **kwargs)
        
        if allocate_ao:
            self._additional_ouput = rdFingerprintGenerator.AdditionalOutput()
            self._additional_ouput.AllocateBitInfoMap()
            self._additional_ouput.AllocateAtomCounts()
            self._additional_ouput.AllocateAtomToBits()
        else:
            self._additional_ouput = None

        self._fingerprint = {
            'bit': partial(self._generator.GetFingerprintAsNumPy, additionalOutput=self._additional_ouput) if allocate_ao else self._generator.GetFingerprintAsNumPy,
            'count': partial(self._generator.GetCountFingerprintAsNumPy, additionalOutput=self._additional_ouput) if allocate_ao else self._generator.GetCountFingerprintAsNumPy,
        }

    def fingerprint(self, mol: Chem.Mol, reaction_center: Iterable[int] = [], output_type: str = 'bit', rc_dist_ub: int = None) -> np.ndarray:
        
        if rc_dist_ub is not None and not reaction_center:
            raise ValueError("If providing upper boudn on distance from reaction center, must also provide reaction center")

        feats = self.featurize_mol(mol, reaction_center)
        
        if rc_dist_ub is not None:
            root_atoms = [
                i for i, ft in enumerate(feats)
                if min(ft[-len(reaction_center):]) <= rc_dist_ub
            ]       
            feats = [self.hash_features(ft) for ft in feats]
            return self._fingerprint[output_type](mol, customAtomInvariants=feats, fromAtoms=root_atoms)

        else:
            feats = [self.hash_features(ft) for ft in feats]
            return self._fingerprint[output_type](mol, customAtomInvariants=feats)

    def get_dai(self, atom: Chem.Atom):
        '''
        Returns Daylight atomic invariants for atom
        '''
        # TODO is atom idx canonical? Do I have to first canonicalize smiles? Will this matter?

        dai = [
            atom.GetDegree(), # Heavy atoms only
            atom.GetTotalValence() - atom.GetTotalNumHs(),
            atom.GetAtomicNum(),
            atom.GetMass(),
            atom.GetFormalCharge(),
            int(atom.IsInRing())
        ]

        return dai

    def hash_features(self, atom_feats: tuple):
        return hash(atom_feats) & 0xFFFFFFFF
    
    def featurize_mol(self, mol: Chem.Mol, reaction_center: list[int] = []) -> list[tuple]:
        '''
        Get atomic features for each atom in a molecule

        Args
        ----
        mol:Mol
            molecule
        reaction_center:list[int]
            List of atom indices of reaction center
            atoms. Should be ordered the same way for a given
            reaction center / operator. If provided, atomic feature
            set includes 
        '''
        feats = []
        for atom in mol.GetAtoms():
            aidx = atom.GetIdx()
            dai = self.get_dai(atom)

            spls = [
                len(Chem.GetShortestPath(mol, aidx, rcidx)) - 1 if aidx != rcidx else 0
                for rcidx in reaction_center
            ]

            feats.append(
                tuple(
                    dai + spls
                )
            )

        return feats
    
    @property
    def bit_info_map(self) -> dict:
        if self._additional_ouput:
            return self._additional_ouput.GetBitInfoMap()
        else:
            return {}
    
    @property
    def atom_counts(self) -> tuple:
        if self._additional_ouput:
            return self._additional_ouput.GetAtomCounts()
        else:
            return tuple()
        
    @property
    def atom_to_bits(self) -> tuple:
        if self._additional_ouput:
            return self._additional_ouput.GetAtomToBits()
        else:
            return tuple()

def tanimoto_similarity(bitvec1: np.ndarray, bitvec2: np.ndarray):
    dot = np.dot(bitvec1, bitvec2)
    return dot / (bitvec1.sum() + bitvec2.sum() - dot)


def calc_molecule_rcmcs(
        mol_rc1,
        mol_rc2,
        patt,
        norm='max'
    ):
    '''
    Args
    ----
    mol_rc1:Tuple[Mol, Tuple[int]]
        1st molecule and tuple of its reaction center atom indices
    mol_rc2:Tuple[Mol, Tuple[int]]
        2nd molecule and tuple of its reaction center atom indices
    patt:str
        Reaction center substructure pattern in SMARTS
    norm:str - Normalization to get an index out of
        prcmcs. 'min' normalizes by # atoms in smaller
        of the two substrates, 'max' by that of the larger
    Returns
    -------
    rcmcs:float
        Reaction center max common substructure score [0, 1]
    '''
    rc_scalar = 100

    def _replace(match):
        atomic_number = int(match.group(1))
        return f"[{atomic_number * rc_scalar}#{atomic_number}"
    
    atomic_sub_patt = r'\[#(\d+)'
    pairs = (mol_rc1, mol_rc2)

    patt = re.sub(atomic_sub_patt, _replace, patt) # Mark reaction center patt w/ isotope number

    # Mark reaction center vs other atoms in substrates w/ isotope number
    for pair in pairs:
        for atom in pair[0].GetAtoms():
            if atom.GetIdx() in pair[1]:
                atom.SetIsotope(atom.GetAtomicNum() * rc_scalar) # Rxn ctr atom
            else:
                atom.SetIsotope(atom.GetAtomicNum()) # Non rxn ctr atom

    cleared, patt = mcs_precheck(mol_rc1, mol_rc2, patt) # Prevents FindMCS default behavior of non-rc-mcs

    if not cleared:
        return 0.0

    # Get the mcs that contains the reaction center pattern
    molecules = [elt[0] for elt in pairs]

    res = rdFMCS.FindMCS(
        molecules,
        seedSmarts=patt,
        atomCompare=rdFMCS.AtomCompare.CompareIsotopes,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchChiralTag=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchValences=True,
        timeout=10
    )

    # Compute prc mcs index
    if res.canceled:
        return 0
    elif norm == 'min':
        return res.numAtoms / min(m.GetNumAtoms() for m in molecules)
    elif norm == 'max':
        return res.numAtoms / max(m.GetNumAtoms() for m in molecules)

def calc_lhs_rcmcs(
        rcts_rc1:Iterable,
        rcts_rc2:Iterable,
        patts:Iterable[str],
        norm:str='max'
    ):
    '''
    Calculates atom-weighted reaction rcmcs score of aligned reactions
    using only reactants, NOT the products of the reaction.

    Args
    -------
    rxn_rc:Iterable of len = 2
        rxn_rc[0]:Iterable[str] - Reactant SMILES, aligned to operator
        rxn_rc[1]:Iterable[Iterable[int]] - innermost iterables have reaction
            center atom indices for a reactant
    patts:Iterable[str]
        SMARTS patterns of reaction center fragments organized
        the same way as rxn_rc[1] except here, one SMARTS string per reactant
    '''
    smiles = [rcts_rc1[0], rcts_rc2[0]]
    rc_idxs = [rcts_rc1[1], rcts_rc2[1]]
    molecules= [[Chem.MolFromSmiles(smi) for smi in elt] for elt in smiles]
    mol_rcs1, mol_rcs2 = [list(zip(molecules[i], rc_idxs[i])) for i in range(2)]
    
    n_atoms = 0
    rcmcs = 0
    for mol_rc1, mol_rc2, patt in zip(mol_rcs1, mol_rcs2, patts):
        rcmcs_i = calc_molecule_rcmcs(mol_rc1, mol_rc2, patt, norm=norm)

        if norm == 'max':
            atoms_i = max(mol_rc1[0].GetNumAtoms(), mol_rc2[0].GetNumAtoms())
        elif norm == 'min':
            atoms_i = min(mol_rc1[0].GetNumAtoms(), mol_rc2[0].GetNumAtoms())
        
        rcmcs += rcmcs_i * atoms_i
        n_atoms += atoms_i

    return rcmcs / n_atoms

def mcs_precheck(mol_rc1, mol_rc2, patt):
    '''
    Modifies single-atom patts and pre-checks ring info
    to avoid giving FindMCS a non-common-substructure which
    results in non-reaction-center-inclusive MCSes
    '''
    if patt.count('#') == 1:
        patt = handle_single_atom_patt(mol_rc1, mol_rc2, patt)
    
    cleared = check_ring_infor(mol_rc1, mol_rc2)

    return cleared, patt

def handle_single_atom_patt(mol_rc1, mol_rc2, patt):
    '''
    Pre-pends wildcard atom and bond to single-atom
    patt if mols share a neighbor w/ common isotope,
    ring membership, & bond type between
    '''
    couples = [set(), set()]
    for i, mol_rc in enumerate([mol_rc1, mol_rc2]):
        mol = mol_rc[0]
        rc_idx = mol_rc[1][0]
        for neighbor in mol.GetAtomWithIdx(rc_idx).GetNeighbors():
            nidx = neighbor.GetIdx()
            nisotope = neighbor.GetIsotope()
            in_ring = neighbor.IsInRing()
            bond_type = mol.GetBondBetweenAtoms(rc_idx, nidx).GetBondType()
            couples[i].add((nisotope, in_ring, bond_type))

    if len(couples[0] & couples[1]) > 0:
        patt = '*~' + patt
    
    return patt

def check_ring_infor(mol_rc1, mol_rc2):
    ''''
    Rejects any mol pair where corresponding
    reaction center atoms have distinct ring membership
    '''
    mol1, mol2 = mol_rc1[0], mol_rc2[0]
    for aidx1, aidx2 in zip(mol_rc1[1], mol_rc2[1]):
        a1_in_ring = mol1.GetAtomWithIdx(aidx1).IsInRing()
        a2_in_ring = mol2.GetAtomWithIdx(aidx2).IsInRing()
        
        if a1_in_ring ^ a2_in_ring:
            return False
        
    return True

if __name__ == '__main__':
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
    print()