from rdkit import Chem
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