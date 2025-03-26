import hydra
from omegaconf import DictConfig
from cgr.inference import ReactantGraph, mol_featurizer
import json
from pathlib import Path
import numpy as np

@hydra.main(version_base=None, config_path='../configs', config_name='infer_mech_subgraphs')
def main(cfg: DictConfig):
    
    # TODO: Iterate over all rxn subsets
    with open(Path(cfg.filepaths.raw_data) / "decarbs.json", 'r') as f:
        rxn_subset = json.load(f)

    n_subgraphs = [{} for _ in range(cfg.max_n)]
    unnormed_fts = []
    for rid, elt in rxn_subset.items():
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

    # Construct binary feature matrix
    f = sum(len(elt) for elt in n_subgraphs)
    n = len(unnormed_fts)
    bfm = np.zeros(shape=(n, f)) # Binary feature matrix
    for i, rxn_fts in enumerate(unnormed_fts):
        for j, jfts in enumerate(rxn_fts):
            jfts = np.array(list(jfts), dtype=int) + j
            bfm[i, jfts] = 1

# TODO: Decide on output format of this script
    np.save(Path(cfg.filepaths.interim_data) / "decarb_bfm.npy", bfm)

if __name__ == '__main__':
    main()