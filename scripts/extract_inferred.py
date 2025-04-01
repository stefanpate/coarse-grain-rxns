import hydra
from omegaconf import DictConfig
from cgr.template import extract_reaction_template
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_str, rc_to_nest
from itertools import chain
from collections import defaultdict

@hydra.main(version_base=None, config_path='../configs', config_name='extract_inferred')
def main(cfg: DictConfig):
    '''
    TODO:
    - Read in bfm
    - Read in subgraph instances
    - Get subgraph freqs
    - Iterate over reactions
    - Get union of subgraphs > freq threshold
    - Extract templates
    - Save templates to CSV
    
    '''

    to_nested_lists = lambda x: [[arr.tolist() for arr in side] for side in x]

    mm = pd.read_parquet(
    Path(cfg.filepaths.raw_data) / cfg.src_file
    )

    templates = defaultdict(list)
    for _, row in mm.iterrows():
        rc = to_nested_lists(row['reaction_center'])
        mech_atoms = to_nested_lists(row['mech_atoms'])
        atoms_to_include = [chain(*elt) for elt in zip(rc[0], mech_atoms)]
        am_smarts = row['am_smarts']
        template = extract_reaction_template(am_smarts, atoms_to_include)
        templates[template].append((row["entry_id"], row['mechanism_id']))

    tmp = []
    for template, ems in templates.items():
        entries, mechs = zip(*ems)
        tmp.append((template, list(entries), list(mechs)))

    df = pd.DataFrame(tmp, columns=["template", "entry_id", "mechanism_id"])
    df.to_csv(Path(cfg.filepaths.processed_data) / "mechanistic_reaction_templates.csv", sep=',')

if __name__ == '__main__':
    main()