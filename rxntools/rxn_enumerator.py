from rdkit import Chem
from rdkit.Chem import AllChem

def run_reactants_with_templates(reaction_smarts: str, substrates: list, templates=None):
    """
    Wrapper around RDKit's RunReactants method to handle stereochemical templates.
    If stereochemical templates are provided, the function performs a substructure match
    before running the reaction. If templates are not provided, it runs the reaction as is.

    Parameters
    ----------
    reaction_smarts: str
        The SMARTS string representing the chemical reaction.

    substrates: list of RDKit Mol
        A list of RDKit molecules (reactants) to feed into the reaction.

    templates: list of RDKit Mol, optional
        A list of RDKit molecules representing the stereochemical templates.
        If None, the reaction is run without performing a substructure match.

    Returns
    -------
    list of tuple
        A list of product sets, where each set is a tuple of RDKit molecules.
    """
    # Convert the reaction SMARTS into a reaction object
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    if rxn is None:
        raise ValueError("Invalid reaction SMARTS string provided.")

    # If templates are provided, perform substructure matching
    if templates:
        # Ensure substrates match the templates
        for substrate, template in zip(substrates, templates):
            if not substrate.HasSubstructMatch(template):
                print(f"No substructure match for substrate: {Chem.MolToSmiles(substrate)}")
                return []  # Return empty list if no match

    # Run the reaction and return the products
    try:
        products = rxn.RunReactants(tuple(substrates))
        return products
    except Exception as e:
        raise RuntimeError(f"Error running reactants: {e}")

