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
	scaling_pol_dir = download_github_data(
		filename="NAME_OF_DATA.zip",
		github_uri="https://github.com/LINK TO SOURCE DATA",
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
	mols = read(scaling_pol_dir/"NAME_OF_DATABASE_FROM_ZIPFILE.xyz",':')

	###########################################################################
	### EVALUATE: SOME CUSTOM CALCULATION
	mol_out = []
	for mol in mols:
		calc = copy(clean_calc)
		mol.calc = calc
		opt = BFGS(mol,maxstep=0.05)
		opt.run(fmax=0.01,steps=100)
		mol_out.append(mol)
	###########################################################################

	### WRITE OUTPUT
	if len(mol_out) > 0:
		write_dir = OUT_PATH/model_name # USE THIS, DO NOT ALTER
		write_dir.mkdir(parents=True, exist_ok=True) # AND THIS, DO NOT ALTER
		write(write_dir/"NAME_OF_DATABASE_FROM_ZIPFILE.xyz", mol_out) # AND THIS, WITH THE CORRECT FILENAME
