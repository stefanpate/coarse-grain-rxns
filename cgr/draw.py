from rdkit import Chem
from rdkit.Chem import Draw


def draw_reaction(rxn: str, sub_img_size: tuple = (300, 200), use_smiles: bool = True) -> str:
    '''
    Draw reaction to svg string

    Args
    -----
    rxn:str
        Reaction SMARTS
    sub_img_size:tuple
        width by height
    use_smiles:bool
        If True, is more explicit about double
        bond location in drawing
    '''
    rxn = Chem.rdChemReactions.ReactionFromSmarts(rxn, useSmiles=use_smiles)
    return Draw.ReactionToImage(rxn, useSVG=True, subImgSize=sub_img_size)

def draw_molecule(smiles: str, size: tuple = (200, 200), hilite_atoms: tuple = tuple(), draw_options: dict = {}) -> str:
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
    draw_options:dict
        Key-value pairs to set fields of 
        rdkit.Chem.Draw.drawOptions object
    '''
    mol = Chem.MolFromSmiles(smiles)
    
    # Catch failed MolFromSmiles
    if mol is None: 
        mol = Chem.MolFromSmiles(smiles, sanitize=False)

    drawer = Draw.MolDraw2DSVG(*size)
    _draw_options = drawer.drawOptions()
    for k, v in draw_options.items():
        try:
            setattr(_draw_options, k, v)
        except Exception as e:
            print(e)

    drawer.DrawMolecule(mol, highlightAtoms=hilite_atoms)
    
    drawer.FinishDrawing()
    img = drawer.GetDrawingText()

    return img

if __name__ == '__main__':
    draw_molecule('CCO')