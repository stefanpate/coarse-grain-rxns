import hydra
from omegaconf import DictConfig
from cgr.rule_writing import extract_reaction_template
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest
from collections import defaultdict
import logging
import numpy as np
from rdkit import Chem

log = logging.getLogger(__name__)

def get_atoms_to_include(rxn: str, lhs_rc: tuple[tuple[int]],  R: int) -> list[tuple[int]]:
    lhs_mols = [Chem.MolFromSmiles(smi) for smi in rxn.split('>>')[0].split('.')]
    atoms_to_include = []
    for mol, rc in zip(lhs_mols, lhs_rc):
        rc = np.array(rc)
        D = Chem.GetDistanceMatrix(mol)
        incl_mask = np.any(D[rc] <= R, axis=0)
        ati = np.where(incl_mask)[0]
        atoms_to_include.append(tuple([int(i) for i in ati]))
    return atoms_to_include

@hydra.main(version_base=None, config_path='../configs', config_name='write_rcr_rules')
def main(cfg: DictConfig):
    if cfg.R == 0:
        return

    min_mapped = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.src_file
    )

    templates = {}
    for _, row in min_mapped.iterrows():
        rc = rc_to_nest(row['template_aidxs'])
        am_smarts = row['am_smarts']
        atoms_to_include = get_atoms_to_include(am_smarts, rc[0], cfg.R)
        try:
            template = extract_reaction_template(rxn=am_smarts, atoms_to_include=atoms_to_include, reaction_center=rc[0])
        except Exception as e:
            log.info(f"Error extracting template for {row["rxn_id"]}: {e}")
            continue
        
        templates[template] = row["rule_id"]

    df = pd.DataFrame([(i, k, v) for i, (k, v) in enumerate(templates.items())], columns=["id", "smarts", "rc_plus_0_id"])
    df.to_csv(f"rc_plus_{cfg.R}_rules.csv", sep=',', index=False)

if __name__ == '__main__':
    main()