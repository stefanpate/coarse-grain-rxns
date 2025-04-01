import hydra
from omegaconf import DictConfig
from cgr.inference import ReactantGraph, MolFeaturizer, atom_featurizer_v0
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import accumulate
from ergochemics.mapping import rc_to_nest

@hydra.main(version_base=None, config_path='../configs', config_name='infer_mech_subgraphs')
def main(cfg: DictConfig):
    
    rxn_subset = pd.read_parquet(Path(cfg.input_path)) # TODO: cfg.rule_id -> actual rule_id, subselect rxn_subset from src file. May need to reset index?
    mol_featurizer = MolFeaturizer(atom_featurizer_v0)
    
    n_subgraphs = [{} for _ in range(cfg.max_n)] # Subgraphs of size n at index n - 1
    unnormed_fts = []
    for _, elt in rxn_subset.iterrows():
        rxn_fts = [{} for _ in range(cfg.max_n)]
        rcts = elt["smarts"].split(">>")[0]
        rc = rc_to_nest(elt["reaction_center"])[0]
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

                tmp.append(
                    [
                        k + so,
                        rxn_subset.loc[i, "id"],
                        rxn_subset.loc[i, "smarts"],
                        rxn_subset.loc[i, "am_smarts"],
                        rxn_subset.loc[i, "reaction_center"],
                        v.tolist(),
                    ]
                )
    
    df = pd.DataFrame(tmp, columns=["subgraph_id", "rxn_id", "smarts", "am_smarts", "reaction_center", "sg_idxs"])
    df.to_parquet(f"{cfg.rule_id}/subgraph_instances.parquet")
    np.save(f"{cfg.rule_id}/bfm.npy", bfm)

if __name__ == '__main__':
    main()