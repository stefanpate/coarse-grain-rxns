import hydra
from omegaconf import DictConfig
from pathlib import Path
import polars as pl
from rxnmapper import RXNMapper
import logging
from tqdm import tqdm

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)


def map_one(rxn_mapper: RXNMapper, smarts: str):
    try:
        res = rxn_mapper.get_attention_guided_atom_maps([smarts])
        return res[0]
    except Exception as e:
        log.warning(f"RXNMapper failed for reaction: {e}")
        return None


def map_batch(rxn_mapper: RXNMapper, smarts_list: list[str]):
    try:
        return rxn_mapper.get_attention_guided_atom_maps(smarts_list)
    except Exception:
        return None


@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="map_with_rxnmapper")
def main(cfg: DictConfig):
    input_path = Path(cfg.filepaths.known) / cfg.input_file
    output_dir = Path(cfg.filepaths.mappings)
    output_path = output_dir / f"mapped_{input_path.stem}_x_rxnmapper.parquet"

    log.info(f"Loading reactions from {input_path}")
    df = pl.read_parquet(input_path)
    if cfg.limit is not None:
        df = df.head(cfg.limit)

    rxn_ids = df["id"].to_list()
    smarts_list = df["smarts"].to_list()
    n_total = len(smarts_list)
    log.info(f"Atom-mapping {n_total} reactions with RXNMapper (batch_size={cfg.batch_size})")

    rxn_mapper = RXNMapper()
    rows = []
    n_skipped = 0

    for start in tqdm(range(0, n_total, cfg.batch_size), desc="Mapping"):
        end = min(start + cfg.batch_size, n_total)
        batch_ids = rxn_ids[start:end]
        batch_smarts = smarts_list[start:end]

        batch_res = map_batch(rxn_mapper, batch_smarts)
        if batch_res is None:
            results = [map_one(rxn_mapper, s) for s in batch_smarts]
        else:
            results = batch_res

        for rxn_id, smarts, r in zip(batch_ids, batch_smarts, results):
            if r is None:
                n_skipped += 1
                continue
            rows.append({
                "rxn_id": rxn_id,
                "smarts": smarts,
                "am_smarts": r["mapped_rxn"],
                "confidence": float(r["confidence"]),
            })

    out_df = pl.DataFrame(
        rows,
        schema={"rxn_id": pl.String, "smarts": pl.String, "am_smarts": pl.String, "confidence": pl.Float64},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)
    log.info(f"Wrote {len(rows)} mapped reactions to {output_path} (skipped {n_skipped} / {n_total})")


if __name__ == "__main__":
    main()
