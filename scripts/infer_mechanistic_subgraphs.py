import hydra
from omegaconf import DictConfig
from cgr.inference import ReactantGraph, mol_featurizer
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Iterable
from itertools import accumulate

def rc_to_str(rc: Iterable[Iterable[Iterable[int]]]) -> str:
    '''
    Convert nested tuple representation of reaction center to string representation.
    '''
    if len(rc) == 1:
        rc = [rc]
        rc.append([])
        rc[-1].append([])
    return ">>".join(
        [
            ";".join(
                [
                    ",".join(
                    [str(aidx) for aidx in mol]
                    )
                    for mol in side
                ]
            )
            for side in rc
        ]
    )

def rc_to_nest(rc: str) -> tuple[tuple[tuple[int]]]:
    '''
    Convert string representation of reaction center to nested tuple representation.
    '''
    return tuple(
        tuple(
            tuple(
                int(aidx) for aidx in mol.split(",") if aidx != ""
            )
            for mol in side.split(";")
        )
        for side in rc.split(">>")
    )

@hydra.main(version_base=None, config_path='../configs', config_name='infer_mech_subgraphs')
def main(cfg: DictConfig):
    
    # TODO: Iterate over all rxn subsets
    with open(Path(cfg.filepaths.raw_data) / "decarbs.json", 'r') as f:
        rxn_subset = json.load(f)
    rule_id = 'decarb' # TODO: Make this a parameter
    
    subgraph_path = Path(f"{rule_id}/subgraphs")
    if not subgraph_path.exists():
        subgraph_path.mkdir(parents=True)

    n_subgraphs = [{} for _ in range(cfg.max_n)] # Subgraphs of size n at index n - 1
    unnormed_fts = []
    rxn_subset = list(rxn_subset.items())
    for rid, elt in rxn_subset:
        rxn_fts = [set() for _ in range(cfg.max_n)]
        rcts = elt["smarts"].split(">>")[0]
        rc = elt["reaction_center"]
        rg = ReactantGraph.from_smiles(rcts=rcts, featurizer=mol_featurizer, rc=rc)
        subgraph_idxs = rg.k_hop_subgraphs(cfg.k)


        for sidxs in subgraph_idxs:
            subgraph = rg.subgraph(sidxs)
            n = len(sidxs) - 1
            nn = len(n_subgraphs[n])
            if nn == 0:
                n_subgraphs[n][0] = subgraph
                rxn_fts[n].add(0)
            else:
                exists = False
                for j, sj in n_subgraphs[n].items():
                    if subgraph == sj:
                        rxn_fts[n].add(j)
                        exists = True
                        break

                if not exists:
                    n_subgraphs[n][nn] = subgraph
                    rxn_fts[n].add(nn)

        unnormed_fts.append(rxn_fts)

    # Construct & save binary feature matrix & examples df
    sidx_offsets = [0] + list(accumulate([len(elt) for elt in n_subgraphs])) # 
    bfm = np.zeros(shape=(len(unnormed_fts), sidx_offsets[-1])) # Binary feature matrix
    tmp = []
    for i, rxn_fts in enumerate(unnormed_fts):
        for j, (so, n_fts) in enumerate(list(zip(sidx_offsets, rxn_fts))):
            n_fts = list(n_fts)
            col_idx = np.array(n_fts, dtype=int) + so
            bfm[i, col_idx] = 1

            for si, k in zip(col_idx, n_fts):
                tmp.append([si, rxn_subset[i][0], rxn_subset[i][1]["smarts"], n_subgraphs[j][k].aidxs, rc_to_str(rxn_subset[i][1]["reaction_center"])])
    
    df = pd.DataFrame(tmp, columns=["subgraph_id", "rxn_id", "smarts", "sidxs", "reaction_center"])
    df.to_parquet(f"{rule_id}/subgraph_examples.parquet")
    np.save(f"{rule_id}/decarb_bfm.npy", bfm)

    # Save subgraphs
    for so, s in zip(sidx_offsets, n_subgraphs):
        for j, sg in s.items():
            sg.save(subgraph_path / f"{rule_id}_{so + j}.npz")


if __name__ == '__main__':
    main()