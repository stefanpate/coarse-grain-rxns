import pytest
from rdkit import Chem
from rxntools import utils
import json

# Define a pytest fixture for cofactors_list
@pytest.fixture
def cofactors_list():
    """
    Pytest fixture to load the cofactors list from the JSON file.
    """
    with open('../data/cofactors.json') as f:
        cofactors_dict = json.load(f)
    return [cofactors_dict[key] for key in cofactors_dict.keys()]

def test_is_isomorphic_ethanol_with_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphism when ethanol is represented with atom-mapped SMARTS & canonical SMILES.
    The consider_stereo parameter is not needed here since ethanol has no stereocenters.
    """
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts("[CH3:1][CH2:2][OH:3]"),
                                mol2 = Chem.MolFromSmiles("CCO"))

def test_is_isomorphic_ethanol_with_atom_mapped_SMARTS_and_non_canonical_SMILES():
    """
    Ensure isomorphism when ethanol is represented with atom-mapped SMARTS & non-canonical SMILES.
    The consider_stereo parameter is not needed here since ethanol has no stereocenters.
    """
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts("[CH3:1][CH2:2][OH:3]"),
                                mol2 = Chem.MolFromSmiles("OCC"))

def test_is_isomorphic_ethanol_without_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphic when ethanol is represented with non-atom-mapped SMARTS & canonical SMILES.
    The consider_stereo parameter is not needed here since ethanol has no stereocenters.
    """
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts("[CH3][CH2][OH]"),
                                mol2 = Chem.MolFromSmiles("CCO"))

def test_is_isomorphic_ethanol_without_atom_mapped_SMARTS_and_non_canonical_SMILES():
    """
    Ensure isomorphic when ethanol is represented with non-atom-mapped SMARTS & non-canonical SMILES.
    The consider_stereo parameter is not needed here since ethanol has no stereocenters.
    """
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts("[CH3][CH2][OH]"),
                                mol2 = Chem.MolFromSmiles("OCC"))

def test_is_isomorphic_nad_plus_with_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphic when NAD(+) is represented with atom-mapped SMARTS & canonical SMILES.
    """
    nad_plus_atom_mapped_smarts_with_stereo = "[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1"
    nad_plus_smiles = "[NH2]C(=O)c1[cH][cH][cH][n+]([C@@H]2O[C@H]([CH2]OP(=O)([OH])OP(=O)([OH])O[CH2][C@H]3O[C@@H](n4[cH]nc5c([NH2])n[cH]nc54)[C@H]([OH])[C@@H]3[OH])[C@@H]([OH])[C@H]2[OH])[cH]1"

    # check if both representations of NAD(+) are equivalent and consider stereochemistry
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts(nad_plus_atom_mapped_smarts_with_stereo),
                                mol2 = Chem.MolFromSmiles(nad_plus_smiles),
                                consider_stereo = True)

def test_is_isomorphic_nadh_with_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphic when NADH is represented with atom-mapped SMARTS & canonical SMILES.
    """
    nadh_atom_mapped_smarts_with_stereo = "[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"
    nadh_smiles = "[NH2]C(=O)C1=[CH]N([C@@H]2O[C@H]([CH2]OP(=O)([OH])OP(=O)([OH])O[CH2][C@H]3O[C@@H](n4[cH]nc5c([NH2])n[cH]nc54)[C@H]([OH])[C@@H]3[OH])[C@@H]([OH])[C@H]2[OH])[CH]=[CH][CH2]1"

    # check if both representations of NADH are equivalent and consider stereochemistry
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts(nadh_atom_mapped_smarts_with_stereo),
                                mol2 = Chem.MolFromSmiles(nadh_smiles),
                                consider_stereo = True)

def test_is_isomorphic_CO2_with_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphic when CO2 is represented with atom-mapped SMARTS & canonical SMILES.
    """
    CO2_atom_mapped_smarts = "[C:23](=[O:24])=[O:27]"
    CO2_smiles = "O=C=O"

    # check if both representations of NADH are equivalent and consider stereochemistry
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts(CO2_atom_mapped_smarts),
                                mol2 = Chem.MolFromSmiles(CO2_smiles))

def test_is_cofactor_CO2_with_canonical_SMILES(cofactors_list):
    """
    Ensure CO2 is flagged as a cofactor when represented with its canonical SMILES string.
    """
    assert utils.is_cofactor(mol = Chem.MolFromSmiles("O=C=O"),
                             cofactors_list = cofactors_list)

def test_is_cofactor_CO2_with_atom_mapped_SMARTS(cofactors_list):
    """
    Ensure CO2 is flagged as a cofactor when represented with its atom-mapped SMARTS string.
    """
    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[C:23](=[O:24])=[O:27]"),
                             cofactors_list = cofactors_list)

def test_is_cofactor_nad_plus_with_atom_mapped_SMARTS(cofactors_list):
    """
    Ensure NAD(+) is flagged as a cofactor when represented with its atom-mapped SMARTS string.
    """
    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1"),
                             cofactors_list = cofactors_list,
                             consider_stereo = True)

def test_is_cofactor_nadh_with_atom_mapped_SMARTS(cofactors_list):
    """
    Ensure NADH is flagged as a cofactor when represented with its atom-mapped SMARTS string.
    """
    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"),
                             cofactors_list = cofactors_list,
                             consider_stereo = True)

def test_remove_stereochemistry_frm_nad_plus():
    """
    Ensure that the utils.remove_stereo(mol) function can successfully remove stereochemistry from NAD(+).
    NAD(+) is represented with its atom-mapped SMILES here.
    We first create a mol object using Chem.MolFromSmarts and then convert the mol object back to SMILES for comparison.
    """
    nad_plus_SMARTS_w_stereo = "[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1"
    nad_plus_mol_w_stereo = Chem.MolFromSmarts(nad_plus_SMARTS_w_stereo)

    nad_plus_mol_wo_stereo = utils.remove_stereo(nad_plus_mol_w_stereo)
    nad_plus_SMARTS_wo_stereo = Chem.MolToSmiles(nad_plus_mol_wo_stereo)
    assert nad_plus_SMARTS_wo_stereo == "[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([CH:12]2[O:13][CH:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][CH:26]3[O:27][CH:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[CH:39]([OH:40])[CH:41]3[OH:42])[CH:43]([OH:44])[CH:45]2[OH:46])[cH:47]1"

def test_remove_stereochemistry_frm_nadh():
    """
    Ensure that the utils.remove_stereo(mol) function can successfully remove stereochemistry from NADH.
    """
    nadh_SMARTS_w_stereo = "[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"
    nadh_mol_w_stereo = Chem.MolFromSmarts(nadh_SMARTS_w_stereo)

    nadh_mol_wo_stereo = utils.remove_stereo(nadh_mol_w_stereo)
    nadh_SMARTS_wo_stereo = Chem.MolToSmiles(nadh_mol_wo_stereo)
    assert nadh_SMARTS_wo_stereo == "[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([CH:12]2[O:13][CH:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][CH:26]3[O:27][CH:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[CH:39]([OH:40])[CH:41]3[OH:42])[CH:43]([OH:44])[CH:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"

def test_remove_stereochemistry_frm_mandelonitrile():
    """
    Ensure that the utils.remove_stereo(mol) function can successfully remove stereochemistry from mandelonitrile.
     """
    mandelonitrile_SMARTS_w_stereo = "[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"
    mandelonitrile_mol_w_stereo = Chem.MolFromSmarts(mandelonitrile_SMARTS_w_stereo)

    mandelonitrile_mol_wo_stereo = utils.remove_stereo(mandelonitrile_mol_w_stereo)
    mandelonitrile_SMARTS_wo_stereo = Chem.MolToSmiles(mandelonitrile_mol_wo_stereo)
    assert mandelonitrile_SMARTS_wo_stereo == "[N:1]#[C:2][CH:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"

def test_remove_stereochemistry_frm_ethanol_AdH_rxn():
    """
    Ensure that the function utils.remove_stereo_frm_rxn() can remove stereochemistry across all species.
    This particular test is for the alcohol dehydrogenation reaction of ethanol to form ethanal.
    """
    rxn_SMARTS_w_stereo = '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'
    rxn_SMARTS_wo_stereo = utils.remove_stereo_frm_rxn(rxn_SMARTS_w_stereo)
    assert rxn_SMARTS_wo_stereo == '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([CH:12]2[O:13][CH:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][CH:26]3[O:27][CH:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[CH:39]([OH:40])[CH:41]3[OH:42])[CH:43]([OH:44])[CH:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([CH:12]2[O:13][CH:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][CH:26]3[O:27][CH:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[CH:39]([OH:40])[CH:41]3[OH:42])[CH:43]([OH:44])[CH:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'

def test_remove_stereochemistry_frm_mandelonitrile_nitrilase_rxn():
    """
    Ensure that the function utils.remove_stereo_frm_rxn() can remove stereochemistry across all species.
    This particular test is for converting mandelonitrile to benzyldehyde via a nitrilase.
    """
    rxn_SMARTS_w_stereo = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    rxn_SMARTS_wo_stereo = utils.remove_stereo_frm_rxn(rxn_SMARTS_w_stereo)
    assert rxn_SMARTS_wo_stereo == '[N:1]#[C:2][CH:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'

def test_remove_stereochemistry_frm_decarboxylation_rxn():
    """
    Ensure that the function utils.remove_stereo_frm_rxn() can remove stereochemistry across all species.
    This particular test is for a decarboxylation reaction.
    """
    rxn_SMARTS_w_stereo = '[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@:20]3([CH:21]=[O:22])[C:23](=[O:24])[O:25][CH3:26].[OH2:27]>>[C:23](=[O:24])=[O:27].[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@H:20]3[CH:21]=[O:22].[OH:25][CH3:26]'
    rxn_SMARTS_wo_stereo = utils.remove_stereo_frm_rxn(rxn_SMARTS_w_stereo)
    assert rxn_SMARTS_wo_stereo == '[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[CH:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[CH:17]2[CH2:18][CH:19]1[C:20]3([CH:21]=[O:22])[C:23](=[O:24])[O:25][CH3:26].[OH2:27]>>[C:23](=[O:24])=[O:27].[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[CH:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[CH:17]2[CH2:18][CH:19]1[CH:20]3[CH:21]=[O:22].[OH:25][CH3:26]'