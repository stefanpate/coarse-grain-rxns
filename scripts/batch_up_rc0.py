from minedatabase.pickaxe import Pickaxe
from copy import deepcopy

def main():
    pk = Pickaxe()
    pk.load_pickled_pickaxe("/projects/b1039/spn1560/coarse-grain-rxns/expansions/2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk")
    n_batches = 20

    for i in range(n_batches):
        batch_pk = Pickaxe()
        batch_pk.reactions = {k: v for j, (k, v) in enumerate(pk.reactions.items()) if j % n_batches == i}
        batch_pk.compounds = {k: v for j, (k, v) in enumerate(pk.compounds.items()) if j % n_batches == i}
        batch_pk.operators = deepcopy(pk.operators)
        batch_pk.coreactants = deepcopy(pk.coreactants)
        batch_pk.targets = deepcopy(pk.targets)
        batch_pk.generation = pk.generation + 1
        print(f"Saving batch {i}: {len(batch_pk.reactions)} reactions, {len(batch_pk.compounds)} compounds")
        batch_pk.pickle_pickaxe(f"/projects/b1039/spn1560/coarse-grain-rxns/expansions/batch_{i}_of_{n_batches}_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk")

    pk.load_pickled_pickaxe(f"/projects/b1039/spn1560/coarse-grain-rxns/expansions/batch_{i}_of_{n_batches}_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk")
    print("Successfully loaded batch")


if __name__ == "__main__":
    main()