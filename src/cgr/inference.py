import networkx as nx
from rdkit import Chem
from pydantic import BaseModel, BeforeValidator, PlainSerializer, ConfigDict
import numpy as np
from typing import Iterable, Callable, Annotated, Any
from itertools import accumulate, chain, permutations, product
from functools import reduce
from pathlib import Path

def ndarray_before_validator(v):
    if not isinstance(v, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(v)} for value {v}")
    return v

def ndarray_serializer(v):
    return v.tolist()

NumpyArray = Annotated[np.ndarray, BeforeValidator(ndarray_before_validator), PlainSerializer(ndarray_serializer, return_type=list)]

class ReactantGraph(BaseModel):
    '''
    Encodes a set of reactants as a graph

    Attributes
    ----------
    V: NumpyArray
        Node (atom) feature matrix (# nodes x # features)
    A: NumpyArray
        Adjacency matrix (weighted w/ bond order)
    aidxs: NumpyArray
        Atom indices, sorted by node feature vectors
    '''
    V: NumpyArray
    A: NumpyArray
    aidxs: NumpyArray
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

        return cls(V=V, A=A, aidxs=np.arange(V.shape[0]))
    
    @classmethod
    def load(cls, filepath: Path | str):
        '''
        Load a ReactantGraph object from a file.

        Args
        ----
        filepath: Path | str
            Path to file containing ReactantGraph ndarrays
            storead as .npz file

        Returns
        -------
        : ReactantGraph
            The ReactantGraph object
        '''
        arrs = np.load(filepath)
        return cls(V=arrs['V'], A=arrs['A'], aidxs=arrs['aidxs'])
    
    def save(self, filepath: Path | str):
        '''
        Save the ReactantGraph object to a .npz file.

        Args
        ----
        filepath: Path | str
            Path to save ReactantGraph object to
        '''
        np.savez(filepath, V=self.V, A=self.A, aidxs=self.aidxs)

    def model_post_init(self, __context: Any) -> None:
        # Sort everything by feature nodes
        srt_nidxs = np.lexsort(self.V.T) # Sorted node idxs
        self.V = self.V[srt_nidxs]
        self.A = self.A[srt_nidxs, :][:, srt_nidxs]
        self.aidxs = self.aidxs[srt_nidxs]

    def subgraph(self, node_idxs: Iterable[int]) -> 'ReactantGraph':
        '''
        Returns subgraph of the reactant graph specified by the node indices provided.

        Args
        ----
        node_idxs: Iterable[int]
            Indices of nodes to include in the subgraph
        Returns
        -------
        : ReactantGraph
            The subgraph
        '''
        if type(node_idxs) is tuple:
            node_idxs = list(node_idxs)
        
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
        
        if this_n != other_n: # Check node cardinality
            return False
        elif not np.array_equal(self.V, other.V): # Check equivalence of sorted nodes
            return False
        elif np.array_equal(self.A, other.A): # Sorted nodes equivalent, check topology
            return True
        else: # Check all permutations of degenerate nodes
            _, idx = np.unique(self.V, axis=0, return_index=True)
            idx = sorted(idx.tolist())

            if idx[-1] != this_n - 1:
                idx += [this_n]

            to_prod = []
            unique_streak = []
            for i in range(len(idx) - 1):
                if idx[i] + 1 == idx[i + 1]:
                    unique_streak.append(idx[i])
                else:    
                    to_prod.append([unique_streak])
                    to_prod.append(list(permutations(range(idx[i], idx[i + 1]))))
                    unique_streak = []

            perms = product(*to_prod)
            perms = [list(chain(*p)) for p in perms]

            for perm in perms:
                if np.array_equal(other.A, self.A[perm, :][:, perm]):
                    return True
        
        return False

def mol_featurizer(mol: Chem.Mol, rc: Iterable[int] = []):
    '''
    Args
    ----
    mol: Chem.Mol
        RDKit molecule object
    rc: Iterable[int] (optional)
        List of atom indices corresponding to reaction center
    '''
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
        atom.GetDegree(), # # heavy atom neighbors
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
    
def amphoteros_ox_state(atom: Chem.Atom):
    '''
    https://amphoteros.com/2013/10/22/counting-oxidation-states/
    '''
    if atom.GetAtomicNum() != 6:
        return -1.0
    else:
        return sum(
            (bond.GetBondTypeAsDouble() - 1.0) + float(bond.GetOtherAtom(atom).GetAtomicNum() != 6)
            for bond in atom.GetBonds()
        )

if __name__ == '__main__':
    smi = 'OC(=O)CCC(N)C(=O)O'
    rc = [(9, 7, 5)]
    rg = ReactantGraph.from_smiles(smi, mol_featurizer, rc)
    print(rg)
    subgaph_idxs = rg.k_hop_subgraphs(3)
    for si in subgaph_idxs:
        print(rg.subgraph(si))

    smi2 = "OC(=O)C(N)CCC(=O)O"
    rc2 = [(0, 1, 3)]
    rg2 = ReactantGraph.from_smiles(smi2, mol_featurizer, rc2)
    print(rg == rg2)

    V = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0]
        ]
    )
    A = np.eye(7)
    A[2, 3] = 1
    A[0, 1] = 1
    P1 = np.eye(7)[[0, 1, 3, 2, 4, 6, 5]]
    P2 = np.eye(7)[[1, 0, 2, 3, 4, 5, 6]]

    rg0 = ReactantGraph(V=V, A=A, aidxs=np.arange(7))
    rg1 = ReactantGraph(V=V, A=P1 @ A @ P1, aidxs=np.arange(7))
    rg2 = ReactantGraph(V=V, A=P2 @ A @ P2, aidxs=np.arange(7))

    assert rg0 == rg1
    assert rg0 != rg2
