from pathlib import Path
from omegaconf import OmegaConf

project_dir = Path(__file__).parents[1]
filepaths = OmegaConf.load(project_dir / "configs" / "filepaths" / "base.yaml")

for k in filepaths.keys():
    filepaths[k] = Path(filepaths[k])