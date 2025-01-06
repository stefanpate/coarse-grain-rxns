import json

def test_cofactors_are_unique():
    with open('../data/cofactors.json', 'r') as f:
        cofactors_dict = json.load(f)

    cofactors_smiles_list = []

    for key in cofactors_dict.keys():
        cofactors_smiles_list.append(cofactors_dict[key])

    assert len(set(cofactors_smiles_list)) == len(cofactors_smiles_list)