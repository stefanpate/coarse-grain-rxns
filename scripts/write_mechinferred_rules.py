import hydra
from omegaconf import DictConfig
from cgr.rule_writing import extract_reaction_template
from cgr.ml import bin_label_to_sep_aidx
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest, get_reaction_center
import logging
from tqdm import tqdm
from cgr.rule_writing import filter_by_pub_date

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../configs', config_name='write_mechinferred_rules')
def main(cfg: DictConfig):

    if cfg.include_stereo:
        predict_set = "mapped_known_reactions_stereo_x_rxnmapper"
    else:
        predict_set = "mapped_known_reactions_x_rc_plus_0_rules"

    # Load min mapped pathway reactions
    log.info("Loading data...")
    aam_rxns = pd.read_parquet(
        Path(cfg.filepaths.mappings) / f"{predict_set}.parquet"
    )

    if cfg.cutoff_date is not None:
        pub_dates = pd.read_parquet(Path(cfg.filepaths.raw_data) / cfg.pub_date_file)
        aam_rxns = filter_by_pub_date(pub_dates, aam_rxns, cfg.cutoff_date, mode="before")

    if not cfg.include_stereo:
        aam_rxns["template_aidxs"] = aam_rxns["template_aidxs"].apply(rc_to_nest)

    # Load predicted mech probas
    if cfg.direct_mcsa_only:
        training_set = f"before_{cfg.cutoff_date}_direct_mcsa_only"
    elif cfg.cutoff_date is not None:
        training_set = f"before_{cfg.cutoff_date}"
    else:
        training_set = "all_data"
    glob_pattern = f"train_{training_set}_predict_{predict_set}_split_*.parquet"
    preds = []
    for fn in (Path(cfg.mech_probas_dir)).glob(glob_pattern):
        log.info(f"Loading: {fn}")
        preds.append(pd.read_parquet(fn))
    if not preds:
        raise FileNotFoundError(f"No mech_probas files matched {Path(cfg.mech_probas_dir) / glob_pattern}")

    pred_df = pd.concat(preds)
    pred_df = pred_df.groupby(["rxn_id", "aidx"]).agg({"probas": "mean"}).reset_index()

    suffix = "_stereo" if cfg.include_stereo else ""

    # Write rules
    log.info("Writing rules...")
    for dt in cfg.decision_thresholds:
        log.info(f"Decision threshold: {dt}")
        templates = {}
        pred_df["y_pred"] = (pred_df["probas"] > dt).astype(int)
        for _, row in tqdm(aam_rxns.iterrows(), total=aam_rxns.shape[0], desc="Extracting templates"):
            am_smarts = row['am_smarts']
            rxn_id = row['rxn_id']

            if cfg.include_stereo:
                try:
                    rc = get_reaction_center(am_smarts, include_stereo=True)
                except Exception as e:
                    log.info(f"Error getting reaction center for {rxn_id}: {e}")
                    continue
            else:
                rc = row['template_aidxs']

            y_pred = pred_df.loc[pred_df["rxn_id"] == rxn_id, "y_pred"].to_numpy()
            atoms_to_include, _ = bin_label_to_sep_aidx(bin_label=y_pred, am_smarts=am_smarts)
            try:
                template = extract_reaction_template(
                    rxn=am_smarts,
                    atoms_to_include=atoms_to_include,
                    reaction_center=rc[0],
                    include_stereo=cfg.include_stereo,
                )
            except Exception as e:
                log.info(f"Error extracting template for {rxn_id}: {e}")
                continue

            if cfg.include_stereo:
                templates[template] = row["confidence"]
            else:
                templates[template] = row["rule_id"]

        if cfg.include_stereo:
            df = pd.DataFrame(
                [(i, k, v) for i, (k, v) in enumerate(templates.items())],
                columns=["id", "smarts", "confidence"],
            )
        else:
            df = pd.DataFrame(
                [(i, k, v) for i, (k, v) in enumerate(templates.items())],
                columns=["id", "smarts", "rc_plus_0_id"],
            )
        _training_set = "_" + training_set if training_set != "all_data" else ""
        df.to_csv(f"mechinferred_dt_{int(dt * 1e3):03d}_rules{_training_set}{suffix}.csv", sep=',', index=False)

if __name__ == '__main__':
    main()
