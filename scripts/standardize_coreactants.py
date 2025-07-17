from ergochemics.standardize import standardize_smiles, fast_tautomerize
from itertools import product
import pandas as pd
from functools import partial
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="standardize_coreactants")
def main(cfg: DictConfig) -> None:

    stdsmi = partial(standardize_smiles, quiet=True, neutralization_method='simple')

    for fn in cfg.fns:
        df = pd.read_csv(f"{cfg.filepaths.raw_data}/{fn}", sep="\t")
        expanded_coreactants = []

        if fn.startswith("paired_"):
            for i, row in df.iterrows():
                coreactant_1 = stdsmi(row["Smiles 1"])
                coreactant_2 = stdsmi(row["Smiles 2"])
                taut_combos = product(
                    fast_tautomerize(coreactant_1),
                    fast_tautomerize(coreactant_2),
                )
                for combo in taut_combos:
                    expanded_coreactants.append(
                        [
                            row["Class 1"],
                            row["Class 2"],
                            combo[0],
                            combo[1],
                            row["Name 1"],
                            row["Name 2"],
                        ]
                    )
            cols = ["class_1", "class_2", "smiles_1", "smiles_2", "name_1", "name_2"]
            new = pd.DataFrame(
                expanded_coreactants, columns=cols
            )
        elif fn.startswith("unpaired_"):
            for i, row in df.iterrows():
                coreactant = stdsmi(row["Smiles"])
                taut_combos = fast_tautomerize(coreactant)
                for combo in taut_combos:
                    expanded_coreactants.append(
                        [
                            row["Class"],
                            combo,
                            row["Name"],
                        ]
                    )
            cols = ["class", "smiles", "name"]
            new = pd.DataFrame(
                expanded_coreactants, columns=cols
            )
        
        new.to_csv(f"{fn.split('_')[0]}.tsv", sep="\t", index=False)

if __name__ == '__main__':
    main()

