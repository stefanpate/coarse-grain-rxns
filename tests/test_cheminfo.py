from cgr.cheminfo import (
    rc_neighborhood,
    MorganFingerprinter,
)
from rdkit import Chem

def test_fingerprinter():
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