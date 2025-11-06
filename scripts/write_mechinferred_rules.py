import hydra
from omegaconf import DictConfig
from cgr.rule_writing import extract_reaction_template
from cgr.ml import bin_label_to_sep_aidx
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../configs', config_name='write_mechinferred_rules')
def main(cfg: DictConfig):

    log.info("Loading data...")
    # Load min mapped pathway reactions
    min_mapped = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.min_mapped
    )

    min_mapped["rxn_id"] = min_mapped["rxn_id"]
    min_mapped["template_aidxs"] = min_mapped["template_aidxs"].apply(rc_to_nest)

    # Load predicted mech probas
    preds = []
    for fn in (Path(cfg.mech_probas_dir)).glob("*.parquet"):
        log.info(f"Loading: {fn}")
        preds.append(pd.read_parquet(fn))

    pred_df = pd.concat(preds)
    pred_df = pred_df.groupby(["rxn_id", "aidx"]).agg({"probas": "mean"}).reset_index()
    pred_df.head()

    # Write rules
    log.info("Writing rules...")
    for dt in cfg.decision_thresholds:
        log.info(f"Decision threshold: {dt}")
        templates = {}
        pred_df["y_pred"] = (pred_df["probas"] > dt).astype(int)
        for _, row in tqdm(min_mapped.iterrows(), total=min_mapped.shape[0], desc="Extracting templates"):
            rc = row['template_aidxs']
            am_smarts = row['am_smarts']
            rxn_id = row['rxn_id']
            y_pred = pred_df.loc[pred_df["rxn_id"] == rxn_id, "y_pred"].to_numpy()
            atoms_to_include, _ = bin_label_to_sep_aidx(bin_label=y_pred, am_smarts=am_smarts)
            try:
                template = extract_reaction_template(rxn=am_smarts, atoms_to_include=atoms_to_include, reaction_center=rc[0])
            except Exception as e:
                log.info(f"Error extracting template for {row["rxn_id"]}: {e}")
                continue
            
            templates[template] = row["rule_id"]

        df = pd.DataFrame([(i, k, v) for i, (k, v) in enumerate(templates.items())], columns=["id", "smarts", "rc_plus_0_id"])
        df.to_csv(f"mechinferred_dt_{int(dt * 100):03d}_rules.csv", sep=',', index=False)

if __name__ == '__main__':
    main()