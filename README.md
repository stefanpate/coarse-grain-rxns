# Coarse-grain reactions

Explores and analyzes ways to coarse-grain enzymatic reactions into a set of rules.

## Getting Started

We recommend using uv to manage dependencies and set up virtual environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
uv python install 3.13 # Install uv
git clone git@github.com:stefanpate/coarse-grain-rxns.git
cd coarse-grain-rxns
uv sync
```

Create a file conf/filepaths/filepaths.yaml based on the structure in TEMPLATE_FILEPATHS.yaml

## Usage

### Writing rules that include reaction center atoms + atoms within R hops (bonds) from the reaction center.

For single R:

```
python scripts/write_rcr_rules.py R=1
```

For multiple Rs:

```
python scripts/write_rcr_rules.py -m R=1,2,3
```

### Writing mechinformed rules (mechanisms derived from [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/))

```
python scripts/write_mechinformed_rules.py
```

### Training a model on mechinformed templates

(Optional) you may run hyperparameter optimization using Optuna. Specify parameter ranges, objective functions and so on in the associated config file hpo.yaml.

```
python scripts/hpo.py
```

Once you have selected hyperparameters and saved the appropriate configuraitons to configs/full, you can train a production model(s). You can either set hyperparameters in the config file templates under subdirectory full or use the notebook mech_classification_performance.ipynb to generate these config files from Optuna HPO studies.

```
python scripts/train_production.py
```

With that production model, predict labels for atom-mapped reactions. To run the inference script, make sure you have the proper config files in the production subdirectory, including model checkpoints.

```
python scripts/infer_mech_labels.py -m
```

The scores output by the production model for atom-mapped reactions can be used to write mechinferred rules.

### Writing mechinferred rules based on model of mechinformed templates

The inference script stores scores (probas) for atom-mapped reactions. These reactions are sourced from [Rhea](https://www.rhea-db.org/) and stored in the schemas specified [here](https://github.com/stefanpate/enz-rxn-data). Atom-mappings are obtained by application of minimal (reaction center only) rules using functions from [this library](https://github.com/stefanpate/ergochemics). 

Using these scores, you can generate SMARTS-encoded reactions rules which can be applied using cheminformatics libraries such as RDkit. You can tune the permisiveness of these rules by varying the decision threshold for inclusion of an atom in a rule template. A default set of decision thresholds for the included production model are specified in the config file write_mechinferred_rules.yaml.

```
python scripts/write_mechinferred_rules.py
```

### Overlaying coreactant roles on rule set (Optional)

This is done so that rules can be used in biosynthesis software, [Pickaxe](https://github.com/tyo-nu/MINE-Database/tree/master), which assumes a default set of available coreactants in a biological context, e.g., NADH/NAD+, ATP, Pi.

```
python scripts/add_coreactant_roles.py mapped_rxns_fn=mapped_known_reactions_x_rc_plus_0_rules.parquet
```

Generic rules, those without special coreactant roles specified, are maintained. Rules with explicit special coreactant rules are added to this and saved to a tsv file.