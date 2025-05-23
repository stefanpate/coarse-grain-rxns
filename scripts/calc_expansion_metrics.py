import hydra
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from minedatabase.pickaxe import Pickaxe
from functools import partial
from DORA_XGB import DORA_XGB

def process_compounds(pk: Pickaxe) -> list[list[int, int, str, str]]:
    data = []
    for k, v in pk.compounds.items():
        
        # Only the "universal" / default currency coreactants have id start with "X"
        # Can use this to filter them out when looking at compound stats
        if v["_id"].startswith("X"):
            continue

        gen = v["Generation"]

        if gen == pk.generation:
            continue

        data.append(
            [
                v["SMILES"],
                len(set(v["Reactant_in"])),
                gen,
                v["_id"]
            ]
        )

    return data

def is_valid_rxn(rxn: dict, starters: dict[str, str]):
    '''
    

    Args
    ----

    '''
    in_linked = False
    out_linked = False

    # Neither starter, (mass source) nor X coreactant (non-mass-contributing source)
    non_sources = [
        c_id for _, c_id in rxn["Reactants"]
        if c_id[0] == 'C' and c_id not in starters
    ]

    # Other requirements for reaction are all sources or
    # only sources required for reaction
    if len(non_sources) == 1 or len(non_sources) == 0:
        in_linked = True

    if any([elt[1][0] != 'X' for elt in rxn["Products"]]):
        out_linked = True

    return in_linked and out_linked

def process_reactions(pk: Pickaxe, dxgb) -> float:
    starters = {}
    for v in pk.compounds.values():
        if v["Type"].startswith("Start"):
            starters[v['_id']] = v["ID"]

    data = []
    for v in pk.reactions.values():
        if not is_valid_rxn(rxn=v, starters=starters):
            continue

        is_feasible = dxgb.predict_label(v["Operator_aligned_smarts"])

        data.append(
            [
                v["Operator_aligned_smarts"],
                is_feasible,
                v['_id'],
                list(v['Operators'])
            ]
        )

    return data

@hydra.main(version_base=None, config_path="../configs", config_name="calc_expansion_metrics")
def main(cfg: DictConfig):

    expansions = []
    for exp in cfg.expansion_fns:
        pk = Pickaxe()
        pk.load_pickled_pickaxe(
            Path(cfg.filepaths.interim_data) / exp
        )
        expansions.append(pk)

    # # Process compounds
    # with ProcessPoolExecutor(max_workers=len(expansions)) as executor:
    #     results = list(
    #         tqdm(
    #             executor.map(process_compounds, expansions, chunksize=1),
    #             total=len(expansions),
    #             desc="Procesing compounds"
    #         )
    #     )
    
    # # Save compounds
    # columns = ["smiles", "fan_out", "gen", "id"] 
    # dfes = []
    # for exp, res in zip(cfg.expansion_fns, results):
    #     df = pd.DataFrame(data=res, columns=columns)
    #     df["expansion"] = exp
    #     dfes.append(df)

    # full_df = pd.concat(dfes)
    # full_df.to_parquet(f"{cfg.expansion_name}_compound_metrics.parquet")

    # Process reactions
    dxgb_ac = DORA_XGB.feasibility_classifier(cofactor_positioning='add_concat')
    process_reactions_ = partial(process_reactions, dxgb=dxgb_ac)
    with ProcessPoolExecutor(max_workers=len(expansions)) as executor:
        results = list(
            tqdm(
                executor.map(process_reactions_, expansions, chunksize=1),
                total=len(expansions),
                desc="Procesing reactions"
            )
        )
    
    # Save reactions
    columns = ["smarts", "dxgb_label", "id", "rules"] 
    dfes = []
    for exp, res in zip(cfg.expansion_fns, results):
        df = pd.DataFrame(data=res, columns=columns)
        df["expansion"] = exp
        dfes.append(df)

    full_df = pd.concat(dfes)
    full_df.to_parquet(f"{cfg.expansion_name}_reaction_metrics.parquet")

if __name__ == "__main__":
    main()