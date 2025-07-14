# Coarse-grain reactions

Explores and analyzes ways to coarse-grain enzymatic reactions into a set of rules.

## Getting Started

We recommend using uv to manage dependencies and set up virtual environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
uv python install 3.13 # Install uv
git clone git@github.com:stefanpate/enz-rxn-data.git
cd enz-rxn-data
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