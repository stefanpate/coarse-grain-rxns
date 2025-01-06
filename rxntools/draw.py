from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor, MolDraw2DSVG
from IPython.display import display, SVG
from typing import Tuple
from copy import deepcopy

def highlight_substructures_in_notebook(substrate_smarts: str,
                                        substructure_smarts: str,
                                        size: Tuple[int, int] = (400, 200),
                                        consider_stereo: bool = False) -> SVG:

    """
    Highlights substructures in a jupyter notebook.
    This function was modified from: https://stackoverflow.com/questions/69735586/how-to-highlight-the-substructure-of-a-molecule-with-thick-red-lines-in-rdkit-as

    Parameters
    ----------
    substrate_smarts: str
        Fully atom-mapped SMARTS string of the substrate molecule.
        This SMARTS representation may or may not include stereochemistry.

    substructure_smarts: str
        Fully atom-mapped SMARTS string of the query substructure.
        This query substructure will be highlighted within the substrate if there is a match.
        This SMARTS representation may or may not include stereochemistry.

    size: Tuple[int, int]
        Resolution size of output image of the substrate & any highlighted substructures.

    consider_stereo: bool

    Returns
    -------
    IPython.core.display.SVG
    The SVG image of the substrate with highlighted substructures.

    """

    # although input to this function is an atom-mapped SMARTS representation of the substrate,
    # we generate an RDKit mol object using Chem.MolFromSmiles instead of Chem.MolFromSmarts,
    # because using Chem.MolFromSmarts will result in single bonds being drawn as dotted lines
    substrate_mol = Chem.MolFromSmiles(substrate_smarts)

    # for generating an RDKit mol object from the query substructure, however,
    # we use Chem.MolFromSMarts
    substructure_mol = Chem.MolFromSmarts(substructure_smarts)

    # compute the 2D coordinates for the substrate's mol object
    # the resulting coordinates will then be stored on each atom of the molecule
    rdDepictor.Compute2DCoords(substrate_mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])

    # we add up substructure matches because later on, highlightAtoms expects only one tuple
    # consequently, the tuples of tuples needs to be merged into a single tuple
    if consider_stereo:
        matches = sum(substrate_mol.GetSubstructMatches(substructure_mol,
                                                        useChirality = True), ())
    else:
        matches = sum(substrate_mol.GetSubstructMatches(substructure_mol,
                                                        useChirality = False), ())

    drawer.DrawMolecule(substrate_mol, highlightAtoms = matches)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')

    return SVG(svg)