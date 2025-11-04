import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import numpy as np
from ergochemics.mapping import rc_to_nest
import logging
from cgr.ml import sep_aidx_to_bin_label, scrub_anonymous_template_atoms
from tqdm import tqdm
import rdkit
from rdkit import Chem
from rdchiral.template_extractor import extract_from_reaction
from collections import defaultdict
from sklearn.metrics import accuracy_score

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

def get_rdchiral_preds(rdchiral_template: str, lhs: str, bin_labels: np.ndarray) -> np.ndarray:
    '''
    Performs substructure matching of rdchiral template against the lhs block molecule
    and returns rdchirals predictions in binary format for the highest scoring of the matches.
    '''
    patt = Chem.MolFromSmarts(rdchiral_template)
    lhs_mol = Chem.MolFromSmiles(lhs)
    if lhs_mol is None or patt is None:
        return None
    matches = lhs_mol.GetSubstructMatches(patt)
    if len(matches) == 0:
        return None
    
    best_y_pred = None
    best_acc = 0
    for match in matches:
        _y_pred = np.zeros(bin_labels.shape, dtype=int)
        _y_pred[list(match)] = 1
        _acc = accuracy_score(bin_labels, _y_pred)

        if _acc > best_acc:
            best_acc = _acc
            best_y_pred = _y_pred
    
    return best_y_pred
    

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="extract_rdchiral")
def main(cfg: DictConfig):

    # Load data
    log.info("Loading & preparing data")
    df = pd.read_parquet(
        Path(cfg.filepaths.mechinformed_mapped_rxns)
    )

    # Prep data
    df["template_aidxs"] = df["template_aidxs"].apply(rc_to_nest)
    df['template_aidxs'] = df.apply(lambda x: scrub_anonymous_template_atoms(x.template_aidxs, x.rule), axis=1) # Scrub anonymous atoms from aidxs
    df["binary_label"] = df.apply(lambda x: sep_aidx_to_bin_label(x.am_smarts, x.template_aidxs)[0], axis=1) # Convert aidxs to binary labels for LHS block mol

    pred_data = []
    rdchiral_templates = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting rdchiral templates and preds"):
        am_smarts = row['am_smarts']
        lhs, rhs = am_smarts.split('>>')
        rxn = {'_id': row['rxn_id'], 'reactants': lhs, 'products': rhs}
        res = extract_from_reaction(rxn)
        try:
            rdchiral_template = ">>".join(res['reaction_smarts'].split('>>')[::-1]) # rdchiral returns in retro direction
        except BaseException as e:
            log.warning(f"rdchiral extraction failed for rxn_id {row['rxn_id']} with error: {e}")
            continue

        y_pred = get_rdchiral_preds(res['reactants'], lhs, row['binary_label'].flatten())
        if y_pred is None:
            log.warning(f"rdchiral matching failed for rxn_id {row['rxn_id']}")
            continue

        assert len(y_pred) == len(row['binary_label'])
        
        for i, (y, yp) in enumerate(zip(row['binary_label'].flatten(), y_pred.flatten())):
            pred_data.append(
                {
                    'rxn_id': row['rxn_id'],
                    'aidx': i,
                    'y': y,
                    'y_pred': yp
                }
            )

        rdchiral_templates[rdchiral_template].append(row['rxn_id'])

    pred_df = pd.DataFrame(pred_data)
    pred_df.to_parquet(
        "rdchiral_predictions.parquet",
        index=False
    )

    template_data = [
        {
            'id': i,
            'smarts': tpl,
            'krids': rdchiral_templates[tpl]
        }
        for i, tpl in enumerate(rdchiral_templates.keys())
    ]
    template_df = pd.DataFrame(template_data)
    template_df.to_csv(
        "rchiral_rules.csv",
        index=False
    )

if __name__ == "__main__":
    main()