import hydra
from omegaconf import DictConfig
from cgr.inference import ReactantGraph, mol_featurizer
import json
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import accumulate
from ergochemics.mapping import rc_to_str

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
        rxn_fts = [{} for _ in range(cfg.max_n)]
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
                rxn_fts[n][0] = subgraph.aidxs
            else:
                exists = False
                for j, sj in n_subgraphs[n].items():
                    if subgraph == sj:
                        rxn_fts[n][j] = subgraph.aidxs
                        exists = True
                        break

                if not exists:
                    n_subgraphs[n][nn] = subgraph
                    rxn_fts[n][nn] = subgraph.aidxs

        unnormed_fts.append(rxn_fts)

    # Construct & save binary feature matrix & examples df
    sidx_offsets = [0] + list(accumulate([len(elt) for elt in n_subgraphs]))
    bfm = np.zeros(shape=(len(unnormed_fts), sidx_offsets[-1])) # Binary feature matrix
    tmp = []
    for i, rxn_fts in enumerate(unnormed_fts):
        for so, n_fts in zip(sidx_offsets, rxn_fts):
            for k, v in n_fts.items():
                bfm[i, k + so] = 1

                tmp.append([k + so, rxn_subset[i][0], rxn_subset[i][1]["smarts"], v.tolist(), rc_to_str([rxn_subset[i][1]["reaction_center"], [[]]])])
    
    df = pd.DataFrame(tmp, columns=["subgraph_id", "rxn_id", "smarts", "sg_idxs", "reaction_center"])
    df.to_parquet(f"{rule_id}/subgraph_instances.parquet")
    np.save(f"{rule_id}/bfm.npy", bfm)

if __name__ == '__main__':
    main()