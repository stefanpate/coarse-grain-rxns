import hydra
from omegaconf import DictConfig
from cgr.rule_writing import filter_by_pub_date
from pathlib import Path
import pandas as pd
import ast
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../configs', config_name='filter_imt_by_date')
def main(cfg: DictConfig):
    min_mapped = pd.read_parquet(
        Path(cfg.filepaths.mappings) / cfg.src_file
    )

    if cfg.cutoff_date is not None:
        pub_dates = pd.read_parquet(Path(cfg.filepaths.raw_data) / cfg.pub_dates_file)
        min_mapped = filter_by_pub_date(pub_dates, min_mapped, cfg.cutoff_date)

    # Map filtered rule_ids to rc_plus_0 ni_ids
    rc_plus_0 = pd.read_csv(Path(cfg.filepaths.input_rules) / cfg.rc_plus_0_rules_file)
    rc_plus_0 = rc_plus_0[rc_plus_0['id'].isin(min_mapped['rule_id'])]
    allowed_ni_ids = set()
    for ni_ids_str in rc_plus_0['ni_ids']:
        allowed_ni_ids.update(ast.literal_eval(ni_ids_str))

    # Filter imt_rules: keep if any ni_id's base (strip last _suffix) is in allowed set
    imt = pd.read_csv(Path(cfg.filepaths.input_rules) / cfg.imt_rules_file)
    mask = imt['ni_ids'].apply(lambda x: any(
        '_'.join(ni_id.split('_')[:-1]) in allowed_ni_ids
        for ni_id in ast.literal_eval(x)
    ))
    imt = imt[mask].reset_index(drop=True)
    imt['id'] = imt.index

    suffix = f"_before_{cfg.cutoff_date}" if cfg.cutoff_date is not None else ""
    imt.to_csv(f"imt_rules{suffix}.csv", sep=',', index=False)
    log.info(f"Saved {len(imt)} rules to imt_rules{suffix}.csv")

if __name__ == '__main__':
    main()
