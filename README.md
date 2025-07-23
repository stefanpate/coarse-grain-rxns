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

### Writing mechinferred rules based on model of mechinformed templates



### Overlaying coreactant roles on rule set (Optional)

This is done so that rules can be used in biosynthesis software, [Pickaxe](https://github.com/tyo-nu/MINE-Database/tree/master), which assumes a default set of available coreactants in a biological context, e.g., NADH/NAD+, ATP, Pi.

```
python scripts/add_coreactant_roles.py mapped_rxns_fn=mapped_known_reactions_x_rc_plus_0_rules.parquet
```

Generic rules, those without special coreactant roles specified, are maintained. Rules with explicit special coreactant rules are added to this and saved to a tsv file.