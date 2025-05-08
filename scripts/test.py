import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
import logging
current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)
@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="hpo")
def main(cfg: DictConfig):
    log.info(f"ntasks: {os.environ.get('SLURM_NTASKS')}")
    log.info(type(os.environ.get('SLURM_NTASKS')))
    log.info(f"mem: {os.environ.get('SLURM_MEM_PER_NODE')}")
    log.info(f"mem-per-cpu: {os.environ.get('SLURM_MEM_PER_CPU')}")

if __name__ == "__main__":
    main()