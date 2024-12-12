from rdkit import Chem
from rdkit.Chem import Draw


def draw_reaction(rxn: str, sub_img_size: tuple = (300, 200), use_smiles: bool = True):    
    rxn = Chem.rdChemReactions.ReactionFromSmarts(rxn, useSmiles=use_smiles)
    return Draw.ReactionToImage(rxn, useSVG=True, subImgSize=sub_img_size)

def draw_molecule(smiles: str, size: tuple = (200, 200), hilite_atoms : tuple = tuple()) -> str:
    '''
    Draw molecule to svg string

    Args
    ----
    mol:str
        Molecule SMILES
    stoich:int
        Stoichiometric coefficient
    size:tuple
        (width, height)
    hilite_atoms:tuple
        Atom indices to highlight
    auto_scl:bool
        If True, scales molecule image width proportional
        to log(# of atoms)
    '''
    mol = Chem.MolFromSmiles(smiles)
    
    # Catch failed MolFromSmiles
    if mol is None: 
        mol = Chem.MolFromSmiles(smiles, sanitize=False)

    drawer = Draw.MolDraw2DSVG(*size)
    drawer.DrawMolecule(mol, highlightAtoms=hilite_atoms)
    
    drawer.FinishDrawing()
    img = drawer.GetDrawingText()

    return img