import hydra
from omegaconf import DictConfig
from cgr.rule_writing import extract_reaction_template, filter_by_pub_date
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest
from collections import defaultdict
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../configs', config_name='write_mechinformed_rules')
def main(cfg: DictConfig):

    stereo_suffix = "_stereo" if cfg.include_stereo else ""
    distilled_mech = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / f"distilled_mech_reactions{stereo_suffix}.parquet"
    )

    if cfg.cutoff_date is not None:
        pub_dates = pd.read_parquet(Path(cfg.filepaths.raw_data) / cfg.pub_dates_file)
        distilled_mech = filter_by_pub_date(pub_dates, distilled_mech, cfg.cutoff_date)

    templates = defaultdict(list)
    for _, row in distilled_mech.iterrows():
        rc = rc_to_nest(row['reaction_center'])
        mech_atoms = rc_to_nest(row['mech_atoms'])
        am_smarts = row['am_smarts']
        template = extract_reaction_template(rxn=am_smarts, atoms_to_include=mech_atoms[0], reaction_center=rc[0], include_stereo=cfg.include_stereo)
        templates[template].append((row["entry_id"], row['mechanism_id']))

    tmp = []
    for i, (template, ems) in enumerate(templates.items()):
        entries, mechs = zip(*ems)
        tmp.append((i, template, list(entries), list(mechs)))

    df = pd.DataFrame(tmp, columns=["id", "smarts", "entry_id", "mechanism_id"])
    suffix = f"_before_{cfg.cutoff_date}" if cfg.cutoff_date is not None else ""
    df.to_csv(f"mechinformed_rules{stereo_suffix}{suffix}.csv", sep=',', index=False)

if __name__ == '__main__':
    main()