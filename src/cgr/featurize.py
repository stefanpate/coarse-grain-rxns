
from rdkit import Chem
from typing import Callable, Iterable
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from functools import partial


class MolFeaturizer:
    def __init__(self, atom_featurizer: Callable[[Chem.Atom], list[int | float]]):
        self.atom_featurizer = atom_featurizer

    def featurize(self, mol: Chem.Mol, rc: Iterable[int] = []) -> np.ndarray:
        '''
        Args
        ----
        mol: Chem.Mol
            RDKit molecule object
        rc: Iterable[int] (optional)
            List of atom indices corresponding to reaction center

        Returns
        -------
        fts: np.ndarray
            Node feature matrix of shape (num_atoms, num_features)
        
        Notes
        -----
        1. Distance to reaction center are always the last n features where n is number of reaction center atoms.
        2. If atom is not connected to an rc atom, distance is set to -1.
        '''
        fts = []
        for atom in mol.GetAtoms():
            aidx = atom.GetIdx()
            local_fts = self.atom_featurizer(atom)
            spls = [
                len(Chem.GetShortestPath(mol, aidx, rcidx)) - 1 if aidx != rcidx else 0
                for rcidx in rc
            ]
            fts.append(local_fts + spls)

        fts = np.array(fts)
        return fts
    
class MorganFingerprinter:
    def __init__(self,
            radius: int,
            length: int,
            mol_featurizer: MolFeaturizer,
            allocate_ao: bool = False,
            **kwargs
        ):
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
        self.mol_featurizer = mol_featurizer

    def fingerprint(self, mol: Chem.Mol, reaction_center: Iterable[int] = [], output_type: str = 'bit', rc_dist_ub: int = None) -> np.ndarray:
        
        if rc_dist_ub is not None and not reaction_center:
            raise ValueError("If providing upper boudn on distance from reaction center, must also provide reaction center")

        feats = self.mol_featurizer.featurize(mol, reaction_center)
        feats = [self.hash_features(tuple(ft.tolist())) for ft in feats]

        if rc_dist_ub is not None:
            root_atoms = [
                i for i, ft in enumerate(feats)
                if min(ft[-len(reaction_center):]) <= rc_dist_ub
            ]       
            return self._fingerprint[output_type](mol, customAtomInvariants=feats, fromAtoms=root_atoms)
        else:
            return self._fingerprint[output_type](mol, customAtomInvariants=feats)

    def hash_features(self, atom_feats: tuple):
        # bitwise AND w/ 0xFFFFFFFF to get 32-bit hash expected by rdkit

        return hash(atom_feats) & 0xFFFFFFFF
    
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

'''
Atom featurizers
'''

def dai(atom: Chem.Atom) -> list[int | float]:
    '''
    Returns Daylight atomic invariants for atom
    '''
    dai = [
        atom.GetDegree(), # Heavy atoms only
        atom.GetTotalValence() - atom.GetTotalNumHs(),
        atom.GetAtomicNum(),
        atom.GetMass(),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
    ]

    return dai

def dai_amphoteros(atom: Chem.Atom) -> list[int | float]:
    atomic_invariants = [
        atom.GetDegree(),
        atom.GetTotalValence() - atom.GetTotalNumHs(),
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
        amphoteros_ox_state(atom)
    ]

    return atomic_invariants

def rule_default(atom: Chem.Atom) -> list[int | float]:
    atomic_invariants = [
        atom.GetDegree(),
        atom.GetTotalValence(),
        atom.GetTotalNumHs(),
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
        z(atom)
    ]

    return atomic_invariants

def z(atom: Chem.Atom) -> float:
    '''
    Returns number of heteroatom neighbors for carbon,
    -1 if atom is not carbon
    '''
    if atom.GetAtomicNum() != 6:
        return -1.0
    else:
        return sum(
            float(bond.GetOtherAtom(atom).GetAtomicNum() != 6)
            for bond in atom.GetBonds()
        )
  
def amphoteros_ox_state(atom: Chem.Atom) -> float:
    '''
    Returns
    -------
    : float
        -1 if atom is not carbon
        + (# pi bonds + # heteroatom neighbors) otherwise

    Notes
    -----
    https://amphoteros.com/2013/10/22/counting-oxidation-states/
    '''
    if atom.GetAtomicNum() != 6:
        return -1.0
    else:
        return sum(
            (bond.GetBondTypeAsDouble() - 1.0) + float(bond.GetOtherAtom(atom).GetAtomicNum() != 6)
            for bond in atom.GetBonds()
        )
