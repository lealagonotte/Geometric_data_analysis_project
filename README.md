# **Studying Cross-Modality matching and Perturbation prediction using labeled Gromow-Wasserstein OT**
Project done for validation of Jean Feydy's course at MVA. We were interested by the following paper:
[Cross-Modality matching and Perturbation prediction using labeled Gromow-Wasserstein OT](https://arxiv.org/abs/2405.00838)

## **Installation**
Let's use the modern package management system for python called [uv](https://github.com/astral-sh/uv)

1) To install uv on your device follow instructions [here](https://github.com/astral-sh/uv)

2) Clone the repo
```bash
git clone --recurse-submodules https://github.com/lealagonotte/Geometric_data_analysis_project.git
```

3) Go to the current repo (Geometric_data_analysis_project) and create a virtual environment
```bash
cd Geometric_data_analysis_project/
uv init
uv venv
```

4) Activtae the virtual environment and add the repo's library 
```bash
source .venv/bin/activate
uv sync
```

## **How to patch Perturb-OT for installation**
Once the virtual env and the repo are setup do the following (follow order).

### Patching `perturbot` package installation
1. Go into Perturb-OT's repo and do all the commands they ask for
```bash
cd Perturb-OT/
cd scvi-tools/
pip install .
cd ../ott/
pip install .
cd ../perturbot
pip install .
cd ..
```

2. At that point `perturbot` is almost empty (only __init__ and utils), to fix that, go into Perturb-OT/perturbot/perturbot, copy all submodules and paste them inside the Perturb-OT/perturbot/build/lib/perturbot folder.
```bash
cp -r perturbot/perturbot/. perturbot/build/lib/perturbot/
```

3. Now uninstall and reininstall `perturbot` specifically:
```bash
uv pip uninstall perturbot
cd perturbot/
uv pip install .
```

At this point perturbot.match should work when imported. However you'll likely hit dependency issues regarding jax and anndata:

### Fixing `jaxlib.xla_extension` error
1. What's happening is some good old mismatch between the code in some depedencies and the updated API of JAX 0.8.0
```bash
uv pip install "jax[cpu]==0.4.36" "jaxlib==0.4.36"
```

That should solve it.

### Fixing `anndata` errors
1. The importing dynamics of anndata slightly changed in recent versions which makes it no longer possible to import read at the very top level so to keep dependency code as is we need to downgrade to an anndata version that supported it:
```bash
uv pip install "anndata==0.10.9"
```

That should be it ! Both perturbot.match and perturbot.predict should be importable now :D !