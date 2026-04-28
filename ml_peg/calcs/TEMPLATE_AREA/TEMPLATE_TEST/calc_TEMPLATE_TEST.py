from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
from ase.optimize import BFGS
import numpy as np
import pytest
import json

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_ae_int_chal_halo(mlip: tuple[str, Any]) -> None:
	"""
	ADD A DESCRIPTION OF THE TEST HERE. 

	IT SHOULD BE CLEAR WHAT CALCULATIONS ARE BEING MADE.

	Parameters
	----------
	mlip
		Name of model use and model to get calculator.
	"""
	model_name, model = mlip
	clean_calc = model.get_calculator()
	
 
	#### DOWNLOAD DATA
	data_dir = download_github_data(
		filename="ORCA_AE_INT_CHAL_HALO.zip",
		github_uri="https://github.com/viktorsvahn/teoroo_ML-PEG/raw/refs/heads/main/data/source",
	)

	#### EVALUATION SCHEME
	"""
	The evaluation should:
	  - Read the downloaded data using ASE
	  - Run an ASE-type calculation using `clean_calc` as a calculator
	  - Write the results to: `write_dir = OUT_PATH/model_name`

	See example below.
	"""

	### READ DATA
	mols = read(data_dir/"DFT_data.xyz",':')

	###########################################################################
	### EVALUATE: SOME CUSTOM CALCULATION
	mol_out = []
	for mol in mols:
		calc = copy(clean_calc)
		mol.calc = calc
		REF_bond_length=mol.get_distance(0,1)
		opt = BFGS(mol,maxstep=0.05)
		opt.run(fmax=0.01,steps=100)
		bond_length=mol.get_distance(0,1)
		mol.info["REF_bond_length"]=REF_bond_length
		mol.info['bond_length']=bond_length
		mol_out.append(mol)
	###########################################################################

	### WRITE OUTPUT
	if len(mol_out) > 0:
		write_dir = OUT_PATH/model_name # USE THIS, DO NOT ALTER
		write_dir.mkdir(parents=True, exist_ok=True) # AND THIS, DO NOT ALTER
		write(write_dir/"EA_INT_CHAL_HALO.xyz", mol_out) # AND THIS, WITH THE CORRECT FILENAME
