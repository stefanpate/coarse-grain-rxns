import networkx as nx
from rdkit import Chem
from pydantic import BaseModel, BeforeValidator, PlainSerializer, ConfigDict
import numpy as np
from typing import Iterable, Callable, Annotated
from itertools import accumulate, chain, permutations, product
from functools import reduce

def ndarray_before_validator(v):
    if not isinstance(v, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(v)} for value {v}")
    return v

def ndarray_serializer(v):
    return v.tolist()

NumpyArray = Annotated[np.ndarray, BeforeValidator(ndarray_before_validator), PlainSerializer(ndarray_serializer, return_type=list)]

class ReactantGraph(BaseModel):
    V: NumpyArray # nodes x features
    A: NumpyArray # nodes x nodes
    aidxs: NumpyArray # atom indices, argsorted by features
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_smiles(cls, rcts: str, featurizer: Callable[[Chem.Mol, Iterable[int]], np.ndarray], rc: Iterable[Iterable[int]] = []):
        '''
        Create a ReactantGraph object from a SMILES string of reactants.
        
        Notes: 
            - Chem.MolFromSmiles() and Chem.CombineMols() each index atoms in the order they
            appear in input SMILES string left to right.

        Args
        ----
        rcts: str
            SMILES string of reactants. Multiple reactants should be separated by '.'
        featurizer: Callable
            Function that takes an RDKit molecule and a list of atom indices
            correpsonding to reaction center and returns a feature vector.
        rc: Iterable[Iterable[int]] (optional)
            List of lists of atom indices corresponding to reaction centers in each
            reactant

        Returns
        -------
        ReactantGraph
            A ReactantGraph object containing the adjacency matrix and feature vector.
        
        
        '''
        sep_mols = [Chem.MolFromSmiles(s) for s in rcts.split('.')]
        aidx_offset = accumulate([mol.GetNumAtoms() for mol in sep_mols])

        if type(rc) is not list:
            rc = list(rc)

        # Translate reaction center indices to combined molecule indices
        for i, mol_rc in enumerate(rc):
            if i == 0:
                rc[i] = list(mol_rc)
            else:
                rc[i] = [idx + aidx_offset[i-1] for idx in mol_rc]

        rc = tuple(chain(*rc))
        rcts = reduce(Chem.CombineMols, sep_mols)
        A = Chem.GetAdjacencyMatrix(rcts, useBO=True)
        V = featurizer(rcts, rc)

        # Sort everything by feature nodes
        aidxs = np.lexsort(V.T)
        V = V[aidxs]
        A = A[aidxs, :][:, aidxs]

        return cls(V=V, A=A, aidxs=aidxs)

    def subgraph(self, node_idxs: Iterable[int]) -> 'ReactantGraph':
        V = self.V[node_idxs]
        A = self.A[node_idxs, :][:, node_idxs]
        aidxs = self.aidxs[node_idxs]

        return ReactantGraph(V=V, A=A, aidxs=aidxs)

    def k_hop_subgraphs(self, k: int) -> set[tuple[int]]:
        '''
        Generate all k-hop subgraphs of the reactant graph.

        Args
        ----
        k: int
            Number of hops to consider

        Returns
        -------
        set[tuple[int]]
            A set of tuples containing the indices of atoms in each subgraph.
        '''
        G = nx.from_numpy_array(self.A)
        subgraphs = set()
        for i in range(k + 1):
            for node in G.nodes:
                subgraphs.add(tuple(nx.ego_graph(G, node, radius=i).nodes))

        return subgraphs

    def __eq__(self, other):
        this_n = self.V.shape[0]
        other_n = other.V.shape[0]
        
        if this_n != other_n:
            return False
        elif not np.array_equal(self.V, other.V):
            return False
        elif np.array_equal(self.A, other.A):
            return True
        else:
            unique, idx = np.unique(self.V, axis=0, return_index=True)
            idx = list(idx) + [len(idx) - 1]

            to_prod = []
            unique_streak = []
            for i in range(len(idx) - 1):
                if idx[i] + 1 == idx[i + 1]:
                    unique_streak.append(idx[i])
                else:
                    to_prod.append([unique_streak])
                    to_prod.append(list(permutations(range(idx[i] + 1, idx[i + 1]))))

            perms = product(*to_prod)
            perms = [list(chain(*p)) for p in perms]

            for perm in perms:
                if np.array_equal(other.A, self.A[perm, :][:, perm]):
                    return True
        
        return False

def mol_featurizer(mol: Chem.Mol, rc: Iterable[int] = []):
    fts = []
    for atom in mol.GetAtoms():
        aidx = atom.GetIdx()
        local_fts = atom_featurizer(atom)
        spls = [
            len(Chem.GetShortestPath(mol, aidx, rcidx)) - 1 if aidx != rcidx else 0
            for rcidx in rc
        ]
        fts.append(local_fts + spls)

    fts = np.array(fts)
    return fts

def atom_featurizer(atom: Chem.Atom):
    atomic_invariants = [
        atom.GetDegree(), # Heavy atoms only
        atom.GetTotalValence() - atom.GetTotalNumHs(),
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
        _get_non_aromatic_c_ox_state(atom)
    ]

    return atomic_invariants

def _get_non_aromatic_c_ox_state(atom: Chem.Atom):
    if atom.GetAtomicNum() != 6 or atom.GetIsAromatic(): # Non-aromatic-C get constant outside range
        return -1.0
    else: # Count heteroatom neighbors, scl by bond degree, sum
        d_oxes = [
            bond.GetBondTypeAsDouble() for bond in atom.GetBonds()
            if bond.GetOtherAtom(atom).GetAtomicNum() != 6
        ]
        return sum(d_oxes)

if __name__ == '__main__':
    smi = 'OC(=O)CCC(N)C(=O)O'
    rc = [(9, 7, 5)]
    rg = ReactantGraph.from_smiles(smi, mol_featurizer, rc)
    print(rg)
    print(rg.k_hop_subgraphs(3))
