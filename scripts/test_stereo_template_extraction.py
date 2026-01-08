import hydra
from omegaconf import DictConfig
from pathlib import Path
import polars as pl
from ergochemics.mapping import get_reaction_center
from rxnmapper import RXNMapper
import logging
from cgr.rule_writing import extract_reaction_template
from tqdm import tqdm
import rdkit
from rdkit import Chem
import json
from rdchiral.main import rdchiralRunText

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="test_stereo_template_extraction")
def main(cfg: DictConfig):

    with open(Path(cfg.filepaths.raw_data) / cfg.test_cases, "r") as f:
        test_cases = json.load(f)

    test_reactions = []
    for case in test_cases:
        if len(case['expected']) == 0:
            continue

        smarts = case['smiles'] + ">>" + case['expected'][0]
        test_reactions.append(smarts)

    rxn_mapper = RXNMapper()
    am_reactions = []
    rcs = []
    templates = []
    res = rxn_mapper.get_attention_guided_atom_maps(test_reactions)
    n_passed = 0
    for case, r in zip(test_cases, res):
        am_smarts = r['mapped_rxn']
        am_reactions.append(am_smarts)
        rc = get_reaction_center(am_smarts, include_stereo=True)
        rcs.append(rc)
        template = extract_reaction_template(rxn=am_smarts, atoms_to_include=rc[0], reaction_center=rc[0], include_stereo=True)
        ltemplate, rtemplate = template.split(">>")
        
        # Wrap left template in parentheses if it contains multiple components. rdhchiral expects this
        if "].[" in ltemplate:
            ltemplate = f"({ltemplate})"

        template = f"{ltemplate}>>{rtemplate}"
        templates.append(template)

        output = rdchiralRunText(template, case['smiles'])
        if output == case['expected']:
            log.info(f"Test passed for case: {case['description']}")
            log.info(f"  Extracted template: {template}")
            n_passed += 1
        else:
            log.error(f"Test FAILED for case: {case['description']}")
            log.error(f"  Extracted template: {template}")
            log.error(f"  Expected: {case['expected']}, but got: {output}")

    log.info(f"Passed {n_passed} out of {len(test_cases)} tests.")

if __name__ == "__main__":
    main()