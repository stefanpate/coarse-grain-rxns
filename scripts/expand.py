from minedatabase.pickaxe import Pickaxe
import hydra
from omegaconf import DictConfig
from pathlib import Path
import logging

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="expand")
def main(cfg: DictConfig):
    pk = Pickaxe(
        coreactant_list=Path(cfg.filepaths.coreactants) /  f"{cfg.coreactants}.tsv",
        rule_list=Path(cfg.filepaths.processed_data) / f"{cfg.rules}.tsv",
        errors=True,
        quiet=True,
        filter_after_final_gen=False,
    )

    pk.load_compound_set(compound_file=Path(cfg.filepaths.starters) / f"{cfg.starters}.csv")

    if cfg.a_plus_b:
        pk.set_starters_as_coreactants()

    pk.transform_all(cfg.processes, cfg.generations) # Expand

    # Save results
    pk.pickle_pickaxe(
        Path(cfg.filepaths.interim_data) / f"{cfg.generations}_steps_{cfg.starters}_rules_{cfg.rules}_aplusb_{cfg.a_plus_b}.pk"
    )

if __name__ == '__main__':
    main()