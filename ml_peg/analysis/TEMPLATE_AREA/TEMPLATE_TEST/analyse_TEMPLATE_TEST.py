"""Analyse scaling_pol benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
import numpy as np
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "TEMPLATE_AREA" / "TEMPLATE_TEST" / "outputs"
OUT_PATH = APP_ROOT / "data" / "TEMPLATE_AREA" / "TEMPLATE_TEST"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of molecule names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """


    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_file = model_dir / "EA_INT_CHAL_HALO.xyz"

            ### ONLY PERFORM ANALYSIS IF OUTPUT EXISTS. KEEP THIS.
            if xyz_file:

                ### READ DATA
                mols = read(xyz_file, index=':')

                for mol in mols:
                    if mol.info['charge'] == 0:
                        system_names.append(mol.info['mol'])
                break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_ea_int_chal_halo_OUTPUT.json",
    title="Electron affinities",
    x_label="Reference EA / eV",
    y_label="EA / eV",
    hoverdata={
        "System": get_system_names(),
    },
)
def electron_affinity() -> dict[str, list]:
    """
    Electron affinity computed from energies of optimized
    charge neutral and minus one charged molecules.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted electron affinities.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_file = model_dir / "EA_INT_CHAL_HALO.xyz"
        
        ### ONLY PERFORM ANALYSIS IF OUTPUT EXISTS. KEEP THIS.
        if not xyz_file:
            continue
            
        # Place to copy individual structure files to app data directory
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        ### READ DATA
        mols = read(xyz_file, index=":")

        #######################################################################
        ### DO ANALYSIS
        model_energies_charge_neutral={}
        model_energies_charged={}
        ref_energies_charge_neutral={}
        ref_energies_charged={}
        for mol in mols:
            if mol.info['charge'] == 0:
                model_energies_charge_neutral.update({mol.info['mol']:mol.get_potential_energy()})
                ref_energies_charge_neutral.update({mol.info['mol']:mol.info['REF_energy']})
                write(structs_dir+f'{mol.info['mol']}.xyz',mol)
            if mol.info['charge'] == -1:
                model_energies_charged.update({mol.info['mol']:mol.get_potential_energy()})
                ref_energies_charged.update({mol.info['mol']:mol.info['REF_energy']})            
            
        # STORE REFERENCE VALUES (IF NEEDED)
        if not ref_stored:
            for k in ref_energies_charge_neutral.keys():
                results["ref"].append( ref_energies_charged[k] - ref_energies_charge_neutral[k] )      
            ref_stored=True
        for k in model_energies_charge_neutral.keys():
            results[model_name].append( model_energies_charged[k] - model_energies_charge_neutral[k] )      
            
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_dist_int_chal_halo_OUTPUT.json",
    title="Bond lenghts",
    x_label="Reference bond length /Å",
    y_label="Model bond length /Å",
    hoverdata={
        "System": get_system_names()*2,
        "Net charge": [0]*len( get_system_names())+[-1]*len( get_system_names())
    },
)
def bond_length() -> dict[str, list]:
    """
    Bond lengths computed from optimized charge neutral molecules.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted bond lengths.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_file = model_dir / "EA_INT_CHAL_HALO.xyz"
        
        ### ONLY PERFORM ANALYSIS IF OUTPUT EXISTS. KEEP THIS.
        if not xyz_file:
            continue          

        ### READ DATA
        mols = read(xyz_file, index=":")

        #######################################################################
        ### DO ANALYSIS
        model_bond_lengths_charge_neutral={}
        ref_bond_lenghts_charge_neutral={}
        model_bond_lengths_charged={}
        ref_bond_lenghts_charged={}
        for mol in mols:
            if mol.info['charge'] == 0:
                model_bond_lengths_charge_neutral.update({mol.info['mol']:mol.info['bond_length']})
                ref_bond_lengths_charge_neutral.update({mol.info['mol']:mol.info['REF_bond_length']})
            if mol.info['charge'] == -1:
                model_bond_lengths_charged.update({mol.info['mol']:mol.info['bond_length']})
                ref_bond_lengths_charged.update({mol.info['mol']:mol.info['REF_bond_length']})            
            
        # STORE REFERENCE VALUES (IF NEEDED)
        if not ref_stored:
            for k in ref_bond_lengths_charge_neutral.keys():
                results["ref"].append( ref_bond_lengths_charge_neutral[k])
            for k in ref_bond_lengths_charge_neutral.keys():
                results["ref"].append( ref_bond_lengths_charged[k])                
            ref_stored=True
        for k in model_bond_lengths_charge_neutral.keys():
            results[model_name].append( model_bond_lengths_charge_neutral[k])
        for k in model_bond_lengths_charge_neutral.keys():
            results[model_name].append( model_bond_lengths_charged[k])
            
    return results



@pytest.fixture
def ea_int_chal_halo_errors(electron_affinity) -> dict[str, float]:
    """
    Mean absolute error in electron affinities.
     
    Parameters
    ----------
    electron_affinity 
        Dictionary of reference and predicted electron affinities.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted electron affinity errors for all models.
    """
    
    results = {}
    for model_name in MODELS:
        if electron_affinity[model_name]:
            results[model_name] = mae(
                electron_affinity["ref"], electron_affinity[model_name]
            )
        else:
            results[model_name] = None

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ea_int_chal_halo_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(ea_int_chal_halo_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all electron affinity metrics.

    Parameters
    ----------
    ea_int_chal_halo_errors
        Mean absolute errors for all systems. 

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": ea_int_chal_halo_errors,
    }


def test_ea_int_chal_halo(metrics: dict[str, dict]) -> None:
    """
    Run ea_int_chal_halo test.

    Parameters
    ----------
    metrics
        All ea_int_chal_halo metrics.
    """
    return
