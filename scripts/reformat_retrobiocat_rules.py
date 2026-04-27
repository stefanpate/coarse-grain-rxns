import re
import logging
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig

log = logging.getLogger(__name__)

STRIP_CHARS = " \t\n\r'\"‘’"


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return s


@hydra.main(version_base=None, config_path='../configs', config_name='reformat_retrobiocat_rules')
def main(cfg: DictConfig):
    src = Path(cfg.filepaths.raw_data) / cfg.src_file
    with open(src) as f:
        data = yaml.safe_load(f)

    rows = []
    for name, entry in data.items():
        smarts_list = (entry or {}).get("smarts") or []
        cleaned = [s.strip(STRIP_CHARS) for s in smarts_list]
        cleaned = [s for s in cleaned if s]
        if not cleaned:
            continue
        slug = slugify(name)
        if len(cleaned) == 1:
            rows.append((cleaned[0], slug))
        else:
            for i, s in enumerate(cleaned):
                rows.append((s, f"{slug}_{i}"))

    df = pd.DataFrame(
        [(i, s, n) for i, (s, n) in enumerate(rows)],
        columns=["id", "smarts", "retrobiocat_name"],
    )
    df.to_csv(cfg.dst_file, index=False)
    log.info(f"wrote {len(df)} rows to {Path.cwd() / cfg.dst_file}")


if __name__ == '__main__':
    main()
