from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Set,Tuple,List
from .utils import is_cofactor, remove_stereo, remove_stereo_frm_rxn

class unmapped_reaction:
    """
    Parameters
    ----------
    rxn_str : str
        Unmapped reaction string of the form "A + B = C + D" or "A.B >> C.D"
    """
    def __init__(self, rxn_str: str):
        self.rxn_str = rxn_str

class mapped_reaction:
    """
    Parameters
    ----------
    rxn_smarts : str
        Mapped reaction SMARTS with complete atom mapping of all species in reaction.

    Attributes
    ----------
    rxn_smarts : str
        Mapped reaction SMARTS with complete atom mapping of all species in reaction.

    Methods
    ----------
    _get_mapped_bonds :
        Get the set of all atom-mapped bonds from a molecule's RDKit mol object.
        This is a static method.

    _get_all_changed_atoms:
        Get all transformed atoms as well as broken and formed bonds from a atom-mapped reaction.

    get_all_changed_atoms:
        Same as above but with the added option to include or exclude cofactors.

    get_template_around_rxn_site:
        Extract a SMARTS template for this reaction with varying radii around the reaction site.
        Templates can be extracted both with and without stereochemistry.
        This is a static method.
    """
    def __init__(self,
                 rxn_smarts: str,
                 include_stereo: bool = True):
        self.rxn_smarts = rxn_smarts

    @staticmethod
    def _get_mapped_bonds(mol: Chem.rdchem.Mol) -> Set[Tuple[int, int, Chem.rdchem.BondType]]:
        """
        Parameters
        ----------
        mol : Chem.rdchem.Mol
            RDKit mol object of a single molecule represented with fully atom-mapped SMARTS.

        Returns
        -------
        bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            A set of mapped bonds for the input molecule.
            Each element in this set is a tuple that corresponds to a mapped bond.
            This tuple itself comprises three elements.
            The first two elements are integers indicating the start & end atom indices of a bond.
            Third element corresponds to the bond type, i.e. single, double or aromatic.
            This bond type is a Chem.rdchem.BondType data type.
        """
        # initialize an empty set to store all bonds
        bonds = set()
        for bond in mol.GetBonds():
            starting_atom = bond.GetBeginAtom()
            ending_atom = bond.GetEndAtom()

            # only consider atoms with mapping when storing mapped bonds
            if starting_atom.GetAtomMapNum() > 0 and ending_atom.GetAtomMapNum() > 0:
                bond_tuple = (starting_atom.GetAtomMapNum(),
                              ending_atom.GetAtomMapNum(),
                              bond.GetBondType())
                bonds.add(bond_tuple)

        return bonds

    def _get_all_changed_atoms(self) -> Tuple[ Set[int],
                                               Set[Tuple[int, int, Chem.rdchem.BondType]],
                                               Set[Tuple[int, int, Chem.rdchem.BondType]] ]:
        """
        Parameters
        ----------
        self

        Returns
        -------
        changed_atoms: Set[int]
            The set atom indices corresponding to all changed atoms across the reaction.

        broken_bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            The set of all bonds that were broken across the reaction.
            Each element in this set is a tuple comprising three elements.
            The first two elements are integers indicating start & end atom indices of a bond.
            The third element corresponds to the bond type, i.e. single, double or aromatic.
            This bond type is a Chem.rdchem.BondType data type.

        formed_bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            Similar to broken_bonds, but for formed bonds.
        """
        # create an RDKit reaction object from a fully atom-mapped reaction smarts string
        rxn = AllChem.ReactionFromSmarts(self.rxn_smarts)
        reactants = rxn.GetReactants()
        products = rxn.GetProducts()

        reactant_bonds = set()
        product_bonds = set()

        # collect all mapped bonds from all reactants involved
        for reactant_mol in reactants:
            reactant_bonds |= self._get_mapped_bonds(mol = reactant_mol)

        # collect all mapped bonds from all products involved
        for product_mol in products:
            product_bonds |= self._get_mapped_bonds(mol = product_mol)

        # identify all broken bonds and formed bonds
        broken_bonds = reactant_bonds - product_bonds # bonds present in reactants but not products
        formed_bonds = product_bonds - reactant_bonds # bonds present in products but not reactants

        # collect atom mapped numbers involved in broken or formed bonds
        changed_atoms = set()
        for bond in broken_bonds.union(formed_bonds):
            changed_atoms.add(bond[0]) # atom map number of the first atom
            changed_atoms.add(bond[1]) # atom map number of the second atom

        return changed_atoms, broken_bonds, formed_bonds

    def get_all_changed_atoms(self,
                              include_cofactors: bool,
                              consider_stereo: bool,
                              cofactors_list: list) -> Tuple[ Set[int],
                                                              Set[Tuple[int, int, Chem.rdchem.BondType]],
                                                              Set[Tuple[int, int, Chem.rdchem.BondType]] ]:
        """
        Parameters
        ----------
        self

        include_cofactors: bool
            Whether to include transformed cofactor atoms in tracking all changed atoms

        consider_stereo: bool
            Whether to consider stereochemistry of cofactors specifically

        cofactors_list : list
            List of cofactor SMILES strings

        Returns
        -------
        changed_atoms: Set[int]
            The set atom indices corresponding to all changed atoms across the reaction.

        include_cofactors: bool
            Whether to include cofactor atoms or not in determining which atoms are transformed

        cofactors_list: list
            List of cofactor SMILES

        broken_bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            The set of all bonds that were broken across the reaction.
            Each element in this set is a tuple comprising three elements.
            The first two elements are integers indicating start & end atom indices of a bond.
            The third element corresponds to the bond type, i.e. single, double or aromatic.
            This bond type is a Chem.rdchem.BondType data type.

        formed_bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            Similar to broken_bonds, but for formed bonds.
        """
        # create an RDKit reaction object from a fully atom-mapped reaction smarts string
        rxn = AllChem.ReactionFromSmarts(self.rxn_smarts)
        reactants = rxn.GetReactants()
        products = rxn.GetProducts()

        reactant_bonds = set()
        product_bonds = set()

        # collect all mapped bonds from all reactants involved
        for reactant_mol in reactants:

            # if cofactors are being tracked too, store all mapped bonds for all reactants
            if include_cofactors:
                reactant_bonds |= self._get_mapped_bonds(mol = reactant_mol)

            # if tracking only changed substrate and product atoms but not cofactor atoms,
            else:
                # check if a reactant is a cofactor first and do nothing if this is true
                if is_cofactor(reactant_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):
                    pass

                # if a reactant is not a cofactor, store its mapped bonds
                else:
                    reactant_bonds |= self._get_mapped_bonds(mol = reactant_mol)

        # collect all mapped bonds from all products involved
        for product_mol in products:

            # if cofactors are being tracked too, store all mapped bonds for all products
            if include_cofactors:
                product_bonds |= self._get_mapped_bonds(mol = product_mol)

            # if tracking only changed substrate and product atoms but not cofactor atoms,
            else:
                # check if a product is a cofactor first and do nothing if this is true
                if is_cofactor(product_mol,
                                cofactors_list = cofactors_list,
                                consider_stereo = consider_stereo):
                    pass

                # if a product is not a cofactor, store its mapped bonds
                else:
                    product_bonds |= self._get_mapped_bonds(mol = product_mol)

        # identify all broken bonds and formed bonds
        broken_bonds = reactant_bonds - product_bonds # bonds present in reactants but not products
        formed_bonds = product_bonds - reactant_bonds # bonds present in products but not reactants

        # collect atom mapped numbers involved in broken or formed bonds
        changed_atoms = set()
        for bond in broken_bonds.union(formed_bonds):
            changed_atoms.add(bond[0]) # atom map number of the first atom
            changed_atoms.add(bond[1]) # atom map number of the second atom

        return changed_atoms, broken_bonds, formed_bonds

    @staticmethod
    def get_template_around_rxn_site(atom_mapped_substrate_smarts: str,
                                     reactive_atom_indices: List[int],
                                     radius: int,
                                     include_stereo: bool) -> str:
        """
        Extract the chemical environment for a given substrate around its specified reaction site.
        The reaction site could comprise a single or even multiple carbon atoms.
        Reaction sites must be specified as a list of atom indices of the transformed atoms.
        The substrate mol

        Parameters
        ----------
        atom_mapped_substrate_smarts: str
            Atom-mapped SMARTS string representing the substrate molecule.

        reactive_atom_indices: List[int]
            List of atom indices of the reactive atoms.

        radius: int
            Radius (in bonds) around each reactive atom to extract the environment

        include_stereo: bool
            Whether to include stereochemistry in the generated SMARTS.

        Returns
        -------
        env_smarts: str
            The SMARTS string for the chemical environment around the reaction site.
        """

        substrate_mol = Chem.MolFromSmarts(atom_mapped_substrate_smarts)
        if substrate_mol is None:
            raise ValueError("Invalid SMARTS string provided.")

        if include_stereo:
            pass
        else:
            substrate_mol = remove_stereo(substrate_mol)

        # initialize a set to store the union of all combined atom environments
        combined_env = set()

        # process each reactive site
        for map_num in reactive_atom_indices:
            # find the atom index corresponding to the atom map number
            rooted_atom_index = None
            for atom in substrate_mol.GetAtoms():
                if atom.GetAtomMapNum() == map_num:
                    rooted_atom_index = atom.GetIdx()
                    break

            if rooted_atom_index is None:
                raise ValueError(f"Atom with map number {map_num} not found!")

            # find the atom environment of the current atom
            env = Chem.FindAtomEnvironmentOfRadiusN(substrate_mol, radius = radius, rootedAtAtom = rooted_atom_index)
            combined_env.update(env)

        # after all combined environments have been collected, convert to a sub-molecule
        combined_submol = Chem.PathToSubmol(substrate_mol, list(combined_env))

        # propagate atom map numbers to the sub-molecule
        for atom in combined_submol.GetAtoms():
            original_atom = substrate_mol.GetAtomWithIdx(atom.GetIdx())
            atom.SetAtomMapNum(original_atom.GetAtomMapNum())

        return Chem.MolToSmarts(combined_submol)

