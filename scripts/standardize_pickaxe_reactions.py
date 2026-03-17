from minedatabase.pickaxe import Pickaxe
from ergochemics.standardize import standardize_mol, hash_reaction
from rdkit import Chem
from functools import lru_cache
from omegaconf import DictConfig
import hydra
import polars as pl
from pathlib import Path
from logging import getLogger

@lru_cache(maxsize=10_000)
def std_smi(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = standardize_mol(mol, neutralization_method="simple")
    smi = Chem.MolToSmiles(mol)
    return smi

def std_rxn(rxn: str) -> str:
    lhs, rhs = [side.split(".") for side in rxn.split(">>")]
    lhs = [std_smi(smi) for smi in lhs]
    rhs = [std_smi(smi) for smi in rhs]
    return ".".join(lhs) + ">>" + ".".join(rhs)

current_dir = Path(__file__).parent.parent.resolve()
logger = getLogger(__name__)

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="standardize_pickaxe_reactions")
def main(cfg: DictConfig):
    '''
    Standardizes pickaxe reactions and saves them to a parquet file with:
    id | am_rxn | std_rxn | rxn_hash
    '''
    pk = Pickaxe()
    pk.load_pickled_pickaxe(Path(cfg.filepaths.expansions) / cfg.expansion)

    rows = []
    failed_ct = 0
    logger.info(f"Standardizing reactions from {cfg.expansion} with {len(pk.reactions)} reactions.")
    for i, v in enumerate(pk.reactions.values()):
        am_rxn = v["am_rxn"]
        try:
            std = std_rxn(am_rxn)
        except Exception:
            logger.warning(f"Failed to standardize reaction {v["_id"]} with am_rxn: {am_rxn}")
            failed_ct += 1
            continue
        rxn_id = hash_reaction(std)
        rows.append({"id": rxn_id, "am_rxn": am_rxn, "std_rxn": std})

        if i % 1000 == 0:
            logger.info(f"Standardized {i} reactions so far with {failed_ct} failures.")

    logger.info(f"Standardized {len(rows)} reactions with {failed_ct} failures.")
    stem = Path(cfg.expansion).stem
    out_path = f"{stem}.parquet"
    pl.DataFrame(rows).write_parquet(out_path)

if __name__ == "__main__":
    main()