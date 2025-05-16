import hydra
from omegaconf import DictConfig
from cgr.rule_writing import extract_reaction_template
from pathlib import Path
import pandas as pd
import numpy as np
from ergochemics.mapping import rc_to_str, rc_to_nest
from itertools import chain
from collections import defaultdict
from rdkit import Chem

@hydra.main(version_base=None, config_path='../configs', config_name='write_freqinformed_rules')
def main(cfg: DictConfig):
    '''
    TODO: maybe iterate over all rule_ids and save one single file?
    
    '''

    sg_insts = pd.read_parquet(
        Path(cfg.filepaths.interim_data) / cfg.rule_id / cfg.subgraph_instances
    )

    bfm = np.load(
        Path(cfg.filepaths.interim_data) / cfg.rule_id / cfg.binary_feature_matrix,
    )

    p1 = bfm.sum(axis=0) / bfm.shape[0]
    sg_insts['reaction_center'] = sg_insts['reaction_center'].apply(rc_to_nest)
    sg_insts['sep_sg_idxs'] = sg_insts['sep_sg_idxs'].apply(rc_to_nest)

    lb = cfg.frequency_lb_scl / bfm.shape[0]
    templates = defaultdict(list)
    for name, gb in sg_insts.groupby(by="rxn_id"):
        am_smarts = gb.iloc[0]['am_smarts']
        reaction_center = gb.iloc[0]['reaction_center']
        n_rcts = len(reaction_center[0])
        
        sg_idxs = [set() for _ in range(n_rcts)]
        for _, row in gb.iterrows():
            if p1[row['subgraph_id']] > lb:
                for i, elt in enumerate(row['sep_sg_idxs'][0]):
                    sg_idxs[i].update(elt)

        template = extract_reaction_template(rxn=am_smarts, atoms_to_include=sg_idxs, reaction_center=reaction_center[0])
        templates[template].append(name)

    df = pd.DataFrame(data=[(i, k, v) for i, (k, v) in enumerate(templates.items())], columns=["id", "smarts", "rxn_ids"])
    df.to_csv(Path(cfg.filepaths.processed_data) / "inferred_rules.csv", sep=',', index=False)

if __name__ == '__main__':
    main()