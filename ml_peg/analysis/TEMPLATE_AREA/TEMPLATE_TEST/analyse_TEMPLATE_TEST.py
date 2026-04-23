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
    ADD A DESCRIPTION OF THE TEST HERE. 

    IT SHOULD BE CLEAR HOW THE ANALYSIS IS BEING MADE.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """


    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_file = model_dir / "NAME_OF_DATABASE_FROM_ZIPFILE.xyz"

            ### ONLY PERFORM ANALYSIS IF OUTPUT EXISTS. KEEP THIS.
            if xyz_file:

                ### READ DATA
                mols = read(xyz_file, index=':')

                ###############################################################
                ### EVALUATE: SOME CUSTOM ANALYSIS
                for mol in mols:
                    pass
                ###############################################################
                break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_TEMPLATE_TEST_OUTPUT.json",
    title="TEMPLATE_TEST OUTPUT DESCRIPTION",
    x_label="Reference SUITABLE LABEL /UNIT",
    y_label="SUITABLE LABEL /UNIT",
    hoverdata={
        "System": get_system_names(),
    },
)
def OUTPUT_PROPERTY() -> dict[str, list]:
    """
    ADD A DESCRIPTION OF THE OUTPUT HERE. 

    IT SHOULD BE CLEAR HOW THE OUTPUT IS BEING COMPUTED.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted OUTPUT PROPERTY.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_file = model_dir / "NAME_OF_DATABASE_FROM_ZIPFILE.xyz"
        
        ### ONLY PERFORM ANALYSIS IF OUTPUT EXISTS. KEEP THIS.
        if not xyz_file:
            continue

        ### READ DATA
        mols = read(xyz_file, index=":")

        #######################################################################
        ### DO ANALYSIS
        for mol in mols:
            pass
            ###################################################################

            # STORE REFERENCE VALUES (IF NEEDED)
            if not ref_stored:
                results["ref"].append(??)            
           
            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)

        ref_stored = True
    return results


@pytest.fixture
def TEMPLATE_TEST_errors(OUTPUT_PROPERTY) -> dict[str, float]:
    """
    ADD A DESCRIPTION OF THE ERROR OUTPUT HERE. 

    IT SHOULD BE CLEAR HOW/WHAT ERRORS ARE COMPUTED.

    Parameters
    ----------
    OUTPUT_PROPERTY ### COULD BE, e.g., TOTAL ENERGIES
        Dictionary of reference and predicted OUTPUT PROPERTY.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted OUTPUT PROPERTY errors for all models.
    """
    results = {}
    for model_name in MODELS:

        #######################################################################
        ### GET ERRORS: THIS IS JUST AN EXAMPLE. USE PREDEFINED: mae, rmse, etc
        if OUTPUT_PROPERTY[model_name]:
            results[model_name] = mae(
                OUTPUT_PROPERTY["ref"], OUTPUT_PROPERTY[model_name]
            )
        else:
            results[model_name] = None
        #######################################################################
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "TEMPLATE_TEST_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(TEMPLATE_TEST_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all TEMPLATE_TEST metrics.

    Parameters
    ----------
    TEMPLATE_TEST_errors
        Mean absolute errors for all systems. ### JUST AN EXAMPLE

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": TEMPLATE_TEST_errors,
    }


def test_TEMPLATE_TEST(metrics: dict[str, dict]) -> None:
    """
    Run TEMPLATE_TEST test.

    Parameters
    ----------
    metrics
        All TEMPLATE_TEST metrics.
    """
    return