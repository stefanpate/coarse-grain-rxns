import hydra
from omegaconf import DictConfig
from cgr.rule_writing import extract_reaction_template
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest
from collections import defaultdict
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../configs', config_name='write_mechinformed_rules')
def main(cfg: DictConfig):

    distilled_mech = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.src_file
    )

    templates = defaultdict(list)
    for _, row in distilled_mech.iterrows():
        rc = rc_to_nest(row['reaction_center'])
        mech_atoms = rc_to_nest(row['mech_atoms'])
        am_smarts = row['am_smarts']
        template = extract_reaction_template(rxn=am_smarts, atoms_to_include=mech_atoms[0], reaction_center=rc[0])
        templates[template].append((row["entry_id"], row['mechanism_id']))

    tmp = []
    for i, (template, ems) in enumerate(templates.items()):
        entries, mechs = zip(*ems)
        tmp.append((i, template, list(entries), list(mechs)))

    df = pd.DataFrame(tmp, columns=["id", "smarts", "entry_id", "mechanism_id"])
    df.to_csv("mechinformed_rules.csv", sep=',', index=False)

if __name__ == '__main__':
    main()