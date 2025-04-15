import networkx as nx
from rdkit import Chem
from pydantic import BaseModel, BeforeValidator, PlainSerializer, ConfigDict
import numpy as np
from typing import Iterable, Callable, Annotated, Any
from itertools import accumulate, chain, permutations, product
from functools import reduce
from pathlib import Path
from collections import defaultdict

class MolFeaturizer:
    def __init__(self, atom_featurizer: Callable[[Chem.Atom], list[int | float]]):
        self.atom_featurizer = atom_featurizer

    def featurize(self, mol: Chem.Mol, rc: Iterable[int] = []):
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

def atom_featurizer_v0(atom: Chem.Atom) -> list[int | float]:
    atomic_invariants = [
        atom.GetDegree(),
        atom.GetTotalValence() - atom.GetTotalNumHs(),
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
        _get_non_aromatic_c_ox_state(atom)
    ]

    return atomic_invariants

def atom_featurizer_v1(atom: Chem.Atom) -> list[int | float]:
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

def atom_featurizer_v2(atom: Chem.Atom) -> list[int | float]:
    atomic_invariants = [
        atom.GetDegree(),
        atom.GetTotalValence(),
        atom.GetTotalNumHs(),
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
        amphoteros_ox_state(atom)
    ]

    return atomic_invariants

def _get_non_aromatic_c_ox_state(atom: Chem.Atom) -> float:
    if atom.GetAtomicNum() != 6 or atom.GetIsAromatic(): # Non-aromatic-C get constant outside range
        return -1.0
    else: # Count heteroatom neighbors, scl by bond degree, sum
        d_oxes = [
            bond.GetBondTypeAsDouble() for bond in atom.GetBonds()
            if bond.GetOtherAtom(atom).GetAtomicNum() != 6
        ]
        return sum(d_oxes)
    
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

def ndarray_before_validator(v):
    if not isinstance(v, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(v)} for value {v}")
    return v

def ndarray_serializer(v):
    return v.tolist()

NumpyArray = Annotated[np.ndarray, BeforeValidator(ndarray_before_validator), PlainSerializer(ndarray_serializer, return_type=list)]

class ReactantGraph(BaseModel):
    '''
    Encodes a set of reactants as a graph. May represent reactants from a specific
    reaction or reactants from a reaction template.

    Attributes
    ----------
    V: NumpyArray
        Node (atom) feature matrix (# nodes x # features)
    A: NumpyArray
        Adjacency matrix (weighted w/ bond order)
    aidxs: NumpyArray | None
        Atom indices, sorted by node feature vectors
    sep_aidxs: NumpyArray | None
        Atom indices for each separate reactant, in order they are entered in the SMILES string.
    rct_idxs: NumpyArray | None
        Indices of reactants for each atom in order they are entered in the SMILES string.
    n_rcts: int | None
        Number of reactants in the full graph. If the ReactantGraph instance is a subgraph 
        of a parent graph, this wil be equal to the number of reactants in the parent graph.
    rcsz: int | None
        Number of atoms in the reaction center
    '''
    V: NumpyArray
    A: NumpyArray
    aidxs: NumpyArray | None = None
    sep_aidxs: NumpyArray | None = None
    rct_idxs: NumpyArray | None = None
    n_rcts: int | None = None
    rcsz: int | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_smiles(cls, rcts: str, featurizer: MolFeaturizer, rc: Iterable[Iterable[int]] = [], exclude: Iterable[int] = []):
        '''
        Create a ReactantGraph object from a SMILES string of reactants.
        
        Notes: 
            - Chem.MolFromSmiles() and Chem.CombineMols() each index atoms in the order they
            appear in input SMILES string left to right.

        Args
        ----
        rcts: str
            SMILES string of reactants. Multiple reactants should be separated by '.'
        featurizer: MolFeaturizer
            Instance of MolFeaturizer to use for featurizing the reactants
        rc: Iterable[Iterable[int]] (optional)
            List of lists of atom indices corresponding to reaction centers in each
            reactant

        Returns
        -------
        ReactantGraph
            A ReactantGraph object containing the adjacency matrix and feature vector.       
        '''
        sep_mols = [Chem.MolFromSmiles(s) for s in rcts.split('.')]
        aidx_offset = [0] + list(accumulate([mol.GetNumAtoms() for mol in sep_mols]))

        if type(rc) is not list:
            rc = list(rc)

        # Translate reaction center indices to combined molecule indices
        for i, mol_rc in enumerate(rc):
            rc[i] = [idx + aidx_offset[i] for idx in mol_rc]

        rc = tuple(chain(*rc)) if len(rc) > 0 else tuple()
        
        # Featurize
        rcts = reduce(Chem.CombineMols, sep_mols)
        A = Chem.GetAdjacencyMatrix(rcts, useBO=True)
        V = featurizer.featurize(rcts, rc)
        
        # Compile indices
        sep_aidxs = []
        rct_idxs = []
        aidxs = []
        for i, mol in enumerate(sep_mols):
            if i not in exclude:
                sep_aidxs.append(np.arange(mol.GetNumAtoms(), dtype=np.int32))
                aidxs.append(np.arange(mol.GetNumAtoms(), dtype=np.int32) + aidx_offset[i])
                rct_idxs.append(np.zeros(shape=(mol.GetNumAtoms()), dtype=np.int32) + i)

        aidxs = np.concatenate(aidxs)
        sep_aidxs = np.concatenate(sep_aidxs)
        rct_idxs = np.concatenate(rct_idxs)

        # Exclude atoms from V and A
        if len(exclude) > 0:
            V = V[aidxs]
            A = A[aidxs, :][:, aidxs]        
        
        return cls(V=V, A=A, aidxs=aidxs, sep_aidxs=sep_aidxs, rct_idxs=rct_idxs, n_rcts=len(sep_mols), rcsz=len(rc))
    
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
        int_fields = {
            'n_rcts': None,
            'rcsz': None,
        }
        arrs = np.load(filepath)
        for k in int_fields.keys():
            int_fields[k] = arrs.get(k)

        return cls(V=arrs['V'], A=arrs['A'], aidxs=arrs.get('aidxs'), sep_aidxs=arrs.get('sep_aidxs'), rct_idxs=arrs.get('rct_idxs'), **int_fields)
    
    def save(self, filepath: Path | str) -> None:
        '''
        Save the ReactantGraph object to a .npz file.

        Args
        ----
        filepath: Path | str
            Path to save ReactantGraph object to
        '''
        to_save = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue
            elif isinstance(v, np.ndarray):
                to_save[k] = v
            elif isinstance(v, int) or isinstance(v, float):
                to_save[k] = np.array([v])
        
        np.savez(filepath, **to_save)

    def model_post_init(self, __context: Any) -> None:
        '''
        Sort everything by node features
        '''
        srt_nidxs = np.lexsort(self.V.T) # Sorted node idxs
        self.V = self.V[srt_nidxs]
        self.A = self.A[srt_nidxs, :][:, srt_nidxs]
        self.aidxs = self.aidxs[srt_nidxs] if self.aidxs is not None else None
        self.sep_aidxs = self.sep_aidxs[srt_nidxs] if self.sep_aidxs is not None else None
        self.rct_idxs = self.rct_idxs[srt_nidxs] if self.rct_idxs is not None else None

    def remove_specific_indexing(self) -> None:
        '''
        Remove indexing specific to a particulare reaction. Sets aidxs, n_rcts, & sep_aidxs
        to None. Normalizes & preserves sep_aidxs and rct_idxsas a convenient way to access 
        connected components. Useful for template extraction.
        '''
        self.aidxs = None
        self.n_rcts = None
        self.sep_aidxs = None
        old2new_cc = {v: u for u, v in enumerate(np.unique(self.rct_idxs))}
        rct_idxs = np.zeros_like(self.rct_idxs, dtype=np.int32)
        for old_idx, new_idx in old2new_cc.items():
            rct_idxs[self.rct_idxs == old_idx] = new_idx

        self.rct_idxs = rct_idxs

    def subgraph(self, node_idxs: list[int]) -> 'ReactantGraph':
        '''
        Returns subgraph of the reactant graph specified by the node indices provided.

        Args
        ----
        node_idxs: list[int]
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
        aidxs = self.aidxs[node_idxs] if self.aidxs is not None else None
        sep_aidxs = self.sep_aidxs[node_idxs] if self.sep_aidxs is not None else None
        rct_idxs = self.rct_idxs[node_idxs] if self.rct_idxs is not None else None

        return ReactantGraph(V=V, A=A, aidxs=aidxs, sep_aidxs=sep_aidxs, rct_idxs=rct_idxs, n_rcts=self.n_rcts, rcsz=self.rcsz)

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
    
    def mcs(self, other: 'ReactantGraph') -> list[tuple[int, int]]:
        """
        Find the maximum common subgraph (MCS) between two ReactantGraph instances.
        
        Args
        ----
        other: ReactantGraph
            The other ReactantGraph to compare with.

        Returns
        -------
        mapping: list[tuple[int, int]]
            A list of tuples where each tuple (i, j) indicates that node i in self
            corresponds to node j in other. If no common subgraph is found, returns an empty list.
        """
        node_colors = ({}, {})
        unique_nodes = []
        for i, rg in enumerate([self, other]):
            for nidx in range(rg.V.shape[0]):
                if len(unique_nodes) == 0:
                    unique_nodes.append(rg.V[nidx])
                    node_colors[i][nidx] = [0]
                else:
                    found = False
                    for j, elt in enumerate(unique_nodes):
                        if np.array_equal(elt, rg.V[nidx]):
                            node_colors[i][nidx] = [j]
                            found = True
                            break
                    
                    if not found:
                        unique_nodes.append(rg.V[nidx])
                        node_colors[i][nidx] = [len(unique_nodes) - 1]

        # Convert aromatic bonds to integer
        A_int = np.where(self.A == 1.5, 4, self.A).astype(np.int16)
        other_A_int = np.where(other.A == 1.5, 4, other.A).astype(np.int16)
        
        return mcsplit(A_G=A_int, A_H=other_A_int, node_labels=node_colors)

def mcsplit(A_G: np.ndarray, A_H: np.ndarray, node_labels: tuple[dict[int, list[int]], dict[int, list[int]]] = tuple(), mapping: set[tuple[int]] = set()) -> list[tuple[int, int]]:
    '''
    Implements McSplit algorithm for finding the maximum common subgraph (MCS) between two graphs.
    Works for undirected graphs with integer weighted adjacency matrices. Graph may have labeled nodes.

    Args
    ----
    A_G: np.ndarray
        Adjacency matrix of the first graph (G).
    A_H: np.ndarray
        Adjacency matrix of the second graph (H).
    node_labels: tuple[dict[int, list[int]], dict[int, list[int]]] (Optional)
        A tuple of two dictionaries where keys are node indices and values are lists of labels for each node.
    mapping: set[tuple[int]] (Optional)
        A set of tuples where each tuple (i, j) indicates that node i in the first graph
        corresponds to node j in the second graph. This is used to keep track of already found mappings.
    Returns
    -------
    mapping: set[tuple[int]]
        A set of tuples where each tuple (i, j) indicates that node i in the first graph
        corresponds to node j in the second graph. This is the maximum common subgraph found.
    
    Notes
    -----
    https://doi.org/10.24963/ijcai.2017/99
    
    '''
    # If no node labels provided, initialize them
    if len(node_labels) == 0:
        node_labels = ({i: [0] for i in range(A_G.shape[0])}, {i: [0] for i in range(A_H.shape[0])})
    
    matches = _get_matching_node_pairs(node_labels)
    matches -= mapping # Remove pairs already added to the mapping
    if not matches:
        return mapping
    
    for pair in matches:
        best_mapping = mapping.union(set([pair])) # Update with current pair

        # Update node labels w/ neighbor info wrt current pair
        new_node_labels = ({}, {})
        for i, A in enumerate((A_G, A_H)):
            for j in range(A.shape[0]):
                new_node_labels[i][j] = node_labels[i][j] + [int(A[pair[i], j])]

        new_mapping = mcsplit(A_G, A_H, new_node_labels, best_mapping)

        if len(new_mapping) > len(best_mapping):
            best_mapping = new_mapping

    return best_mapping

def _get_matching_node_pairs(node_labels: tuple[dict[int, int], dict[int, int]]) -> list[tuple[int, int]]:
    """
    Helper function to find matching node pairs between two node label dictionaries
    corresponding to two graphs.
    
    Args
    ----
    node_labels: tuple[dict[int, int], dict[int, int]]
        A tuple of two dictionaries where keys are node indices and values are their labels.

    Returns
    -------
    set[tuple[int, int]]
        A list of tuples where each tuple (i, j) indicates that node i in the first graph
        corresponds to node j in the second graph.
    """
    matches = set()
    for i, label_i in node_labels[0].items():
        for j, label_j in node_labels[1].items():
            if label_i == label_j:
                matches.add((i, j))
    return matches

def get_stereotypical_molecules(smarts_subset: list[str]) -> tuple[tuple[int], tuple[int]]:
    '''
    Identify stereotypical molecules from reactions, i.e.,
    those that are always there.

    Args
    ----
    smarts_subset: list
        SMARTS strings of the reactions to be filtered. They must have 
        reactants and products aligned to the roles in their shared minimal rule.

    Returns
    -------
    : tuple[tuple[int], tuple[int]]
        Indices of stereotypical rcts and pdts 
    '''
    lhs_unique = defaultdict(set)
    rhs_unique = defaultdict(set)
    for j, smarts in enumerate(smarts_subset):
        rcts, pdts = [side.split('.') for side in smarts.split('>>')]

        n_rcts, n_pdts = len(rcts), len(pdts)

        if j == 0:
            last_n_rcts, last_n_pdts = n_rcts, n_pdts

        if n_rcts != last_n_rcts or n_pdts != last_n_pdts:
            raise(ValueError("All reactions must have the same number of reactants and products"))

        for i, rct in enumerate(rcts):
            lhs_unique[i].add(rct)

        for i, pdt in enumerate(pdts):
            rhs_unique[i].add(pdt)

        last_n_rcts, last_n_pdts = n_rcts, n_pdts

    lhs_stereotypical = tuple(i for i, rcts in lhs_unique.items() if len(rcts) == 1)
    rhs_stereotypical = tuple(i for i, pdts in rhs_unique.items() if len(pdts) == 1)

    return (lhs_stereotypical, rhs_stereotypical)
    
if __name__ == '__main__':
    smi = 'OC(=O)CCC(N)C(=O)O'
    rc = [(9, 7, 5)]
    mol_featurizer = MolFeaturizer(atom_featurizer_v1)
    rg = ReactantGraph.from_smiles(smi, mol_featurizer, rc)
    print(rg)
    rg.remove_specific_indexing()  # Remove indexing specific to a particular reaction
    rg.save("test_rg.npz")
    rg_loaded = ReactantGraph.load("test_rg.npz")
    print("Loaded ReactantGraph:", rg_loaded)
    assert rg == rg_loaded, "Loaded ReactantGraph does not match the original."

    smi3 = 'CCC(N)C(=O)O'
    rc3 = [(6, 4, 2)]
    mol_featurizer = MolFeaturizer(atom_featurizer_v1)
    rg3 = ReactantGraph.from_smiles(smi3, mol_featurizer, rc3)

    mcs = rg.mcs(rg3)
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


    smi2 = "OC(=O)C(N)CCC(=O)O.CCO"
    rc2 = [(0, 1, 3), (0, 1)]
    rg2 = ReactantGraph.from_smiles(smi2, mol_featurizer, rc2, exclude=[0])
    print(rg2)
