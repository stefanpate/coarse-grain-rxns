import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from minedatabase.pickaxe import Pickaxe
from ergochemics.mapping import get_reaction_center
from cgr.ml import sep_aidx_to_bin_label
import numpy as np
from functools import partial
from rdkit import Chem
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_compounds(pk: Pickaxe) -> list[list[int, int, str, str]]:
    data = []
    for v in tqdm(pk.compounds.values(), desc="Processing compounds", total=len(pk.compounds)):
        
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
    Returns false if reaction only default currency 'X'
    coreactants
    
    Args
    ----
    rxn:dict
        Pickaxe reaction entry
    starters:dict
        Starting compounds {_id: ID} where _id is a hash
        id and ID is a user provided name
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

def process_reaction(v: dict) -> list:
    if not is_valid_rxn(rxn=v, starters=starters):
        return []

    # Important to use am smarts for mol and fp creation since this is where you
    # get the correct reaction center from and am_smarts and Operator_aligned_smarts
    # are not guaranteed to be the same out of Pickaxe
    query_smarts = v["Operator_aligned_smarts"]
    query_am_smarts = v["am_rxn"]
    query_lhs_mol = Chem.MolFromSmiles(query_am_smarts.split('>>')[0])

    if query_lhs_mol is None:
        # TODO: figure out logging with multiprocessing
        # log.warning(f"Invalid query LHS molecule for reaction {v['_id']}: {query_am_smarts}")
        return []

    try:
        query_lhs_block_rc = get_lhs_block_rc(query_am_smarts)
    except Exception as e:
        # TODO: figure out logging with multiprocessing
        # log.error(f"Error getting reaction center for {v['_id']}: {query_am_smarts}. Error: {e}")
        return []

    rules = set([int(elt.split('_')[0]) for elt in v["Operators"]])
    analogues = mapped_rxns.loc[mapped_rxns.rule_id.isin(rules)]

    if analogues.empty:
        max_sim = 0.0
        nearest_kr = ''
        nearest_krid = ''
    else:
        mfps = np.vstack(analogues["mfp"])
        query_mfp = _fingerprint(
            mol=query_lhs_mol,
            reaction_center=query_lhs_block_rc
        ).reshape(-1, 1)
        sims = np.matmul(mfps, query_mfp) / (np.linalg.norm(mfps, axis=1).reshape(-1, 1) * np.linalg.norm(query_mfp))
        sims = sims.reshape(-1,)
        max_sim = float(np.max(sims))
        max_idx = int(np.argmax(sims))
        nearest_kr = analogues.iloc[max_idx].smarts
        nearest_krid = analogues.iloc[max_idx].rxn_id

    is_feasible = dxgb.predict_label(query_smarts)

    return [
        v['_id'],
        query_smarts,
        query_am_smarts,
        is_feasible,
        max_sim,
        nearest_kr,
        nearest_krid,
        list(v['Operators'])
    ]

def rxn_proc_initializer(cfg: DictConfig, _starters: dict[str, str]):
    print("Initializing reaction processing")
    global dxgb, mfper, _fingerprint, mapped_rxns, starters

    starters = _starters

    print("Loading mapped rxns ", Path(cfg.filepaths.raw_data) / cfg.mapped_rxns)
    mapped_rxns = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.mapped_rxns
    )

    print("Instantiating fingerprinter and dxgb model")
    dxgb = instantiate(cfg.dxgb)
    mfper = instantiate(cfg.mfper)
    _fingerprint = partial(mfper.fingerprint, rc_dist_ub=cfg.rc_dist_ub)

    print("Calculating reaction centers and morgan fps for all mapped reactions")
    mapped_rxns["reaction_center"] = mapped_rxns["am_smarts"].apply(get_lhs_block_rc)
    mapped_rxns["mol"] = mapped_rxns["smarts"].apply(lambda x : Chem.MolFromSmiles(x.split('>>')[0]))
    mapped_rxns["mfp"] = mapped_rxns.apply(lambda x : _fingerprint(x.mol, x.reaction_center), axis=1)

def get_lhs_block_rc(am_smarts: str) -> list[int]:
    rc = get_reaction_center(am_smarts)
    block_rc = [np.flatnonzero(elt) for elt in sep_aidx_to_bin_label(am_smarts, rc)]
    lhs_block_rc = [int(elt) for elt in block_rc[0]]
    return lhs_block_rc

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="calc_expansion_metrics")
def main(cfg: DictConfig):

    print("Loading expansion ", Path(cfg.filepaths.expansions) / cfg.expansion)
    expansion = Pickaxe()
    expansion.load_pickled_pickaxe(
        Path(cfg.filepaths.expansions) / cfg.expansion
    )

    # Collect starters
    _starters = {}
    for v in expansion.compounds.values():
        if v["Type"].startswith("Start"):
            _starters[v['_id']] = v["ID"]

    # Process compounds
    proc_cpds = process_compounds(expansion)
    
    # Save compound metrics
    print("Saving compound metrics")
    columns = ["smiles", "fan_out", "gen", "id"] 
    df = pd.DataFrame(data=proc_cpds, columns=columns)
    df["expansion"] = cfg.expansion
    df.to_parquet(f"{cfg.expansion}_compound_metrics.parquet")

    # Process reactions
    with ProcessPoolExecutor(max_workers=cfg.processes, initializer=rxn_proc_initializer, initargs=(cfg, _starters)) as executor:
        results = list(
            tqdm(
                executor.map(process_reaction, expansion.reactions.values(), chunksize=100),
                total=len(expansion.reactions),
                desc="Procesing reactions"
            )
        )

    # Save reaction metrics
    print("Saving reaction metrics")
    columns = ["id", "smarts", "am_smarts", "dxgb_label", "max_rxn_sim", "nearest_analogue", "nearest_analogue_id", "rules"] 
    df = pd.DataFrame(data=[elt for elt in results if len(elt) != 0], columns=columns)
    df["expansion"] = cfg.expansion
    df.to_parquet(f"{cfg.expansion}_reaction_metrics.parquet")

if __name__ == "__main__":
    main()