conda env remove -n "wandering_env_conda"
conda env create -n "wandering_env_conda" -f env_setup/wi.yml

## Install new environment.
python3 -m ipykernel install --user --name csci1470 --display-name "wi-env (3.10)"
