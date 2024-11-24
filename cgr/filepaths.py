from omegaconf import OmegaConf
from pathlib import Path

CONFIGS = Path(__file__).parents[1] / "configs"
filepaths = OmegaConf.load(CONFIGS / "filepaths.yaml")