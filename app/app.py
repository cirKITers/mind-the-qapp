from layouts.app_page_layout import (
    sidebar_top,
    sidebar_bottom,
    content,
    DEFAULT_ANSATZ,
    DEFAULT_DATA_REUPLOAD,
    DEFAULT_N_LAYERS,
    DEFAULT_N_QUBITS,
    DEFAULT_SEED,
)

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html, callback, State, dcc, html
from typing import Any, Dict, Optional
import sys

from qml_essentials.ansaetze import Ansaetze
from layouts.app_page_layout import (
    DEFAULT_N_QUBITS,
    DEFAULT_N_LAYERS,
    DEFAULT_SEED,
    DEFAULT_DATA_REUPLOAD,
    DEFAULT_ANSATZ,
)

import logging


app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], use_pages=True
)


sidebar = html.Div(
    [
        dcc.Store(id="main-storage", storage_type="session"),
        html.Div(
            [
                html.H1(
                    "Mind",
                ),
                html.H2(
                    "the",
                    style={
                        "padding-left": "6px",
                    },
                ),
                html.Span(
                    [
                        html.Img(
                            src="assets/underground-sign.svg",
                            width="42",
                            style={
                                "display": "inline-block",
                                "padding-right": "5px",
                                "padding-bottom": "20px",
                            },
                            className="rotate45",
                        ),
                        html.H1("App", style={"display": "inline-block"}),
                    ]
                ),
            ],
            className="infoBox",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(page["name"], href=page["relative_path"], active="exact")
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
            fill=False,
        ),
        html.Div(
            [
                html.Div(
                    [
                        dbc.Label("# of Qubits (0-10)"),
                        dbc.Input(
                            type="number",
                            min=0,
                            max=10,
                            step=1,
                            value=DEFAULT_N_QUBITS,
                            id="main-qubits-input",
                        ),
                    ],
                    className="numeric-input",
                ),
                html.Div(
                    [
                        dbc.Label("# of Layers (0-10)"),
                        dbc.Input(
                            type="number",
                            min=0,
                            max=10,
                            step=1,
                            value=DEFAULT_N_LAYERS,
                            id="main-layers-input",
                        ),
                    ],
                    className="numeric-input",
                ),
                html.Div(
                    [
                        dbc.Label("Circuit Type"),
                        dbc.Select(
                            options=[
                                {
                                    "label": fct.__name__.replace("_", " ").title(),
                                    "value": fct.__name__,
                                }
                                for fct in Ansaetze.get_available()
                            ],
                            placeholder=DEFAULT_ANSATZ,
                            required=True,
                            id="main-circuit-ident-select",
                        ),
                    ],
                    className="numeric-input",
                ),
                html.Div(
                    [
                        dbc.Label("Data-Reupload"),
                        dbc.Switch(
                            id="main-dru-switch",
                            value=DEFAULT_DATA_REUPLOAD,
                            className="fs-4",
                        ),
                    ],
                ),
                # html.Div(
                #     [
                #         dbc.Label("Trainable Freqs."),
                #         dbc.Switch(id="switch-tffm", value=False),
                #     ],
                # ),
                html.Div(
                    [
                        dbc.Label("Seed"),
                        dbc.Input(
                            type="number",
                            min=100,
                            max=999,
                            step=1,
                            value=DEFAULT_SEED,
                            id="main-seed-input",
                        ),
                    ],
                    className="numeric-input",
                ),
            ],
            className="preferencesBox",
        ),
        html.Hr(),
        html.Div(
            [
                dbc.Spinner(
                    [html.H6("", id="main-loading-state")],
                    color="primary",
                    type="grow",
                    id="main-loading-spinner",
                )
            ],
            className="spinnerBox",
        ),
    ],
    className="sidebar",
    id="page-sidebar",
)

sidebar_page_elements = dbc.Nav(
    [
        dbc.NavLink(page["name"], href=page["relative_path"], active="exact")
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    fill=False,
)
sidebar_elements = sidebar_top + [sidebar_page_elements] + sidebar_bottom
sidebar = html.Div(sidebar_elements, className="sidebar", id="page-sidebar")


@callback(
    Output("main-storage", "data"),
    [
        Input("main-qubits-input", "value"),
        Input("main-layers-input", "value"),
        Input("main-circuit-ident-select", "value"),
        Input("main-dru-switch", "value"),
        # Input("switch-tffm", "value"),
        Input("main-seed-input", "value"),
    ],
    State("main-storage", "data"),
    # prevent_initial_call=True,
)
def on_preference_changed(
    number_qubits: int,
    number_layers: int,
    circuit_type: int,
    data_reupload: bool,
    seed: int,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Updates the data dict with new values from the preferences.

    Args:
        number_qubits: The number of qubits from the user's input.
        number_layers: The number of layers from the user's input.
        circuit_type: The circuit identifier chosen by the user.
        data_reupload: Whether data reupload is enabled or not.
        seed: The seed for the data generation.
        data: The data dict to update. If None, creates a new dict.

    Returns:
        The updated data dict.
    """
    # Give a default data dict with 0 clicks if there's no data.
    data = data or {}
    data["number_qubits"] = (
        max(min(number_qubits, 10), 0)
        if number_qubits is not None
        else DEFAULT_N_QUBITS
    )
    data["number_layers"] = (
        max(min(number_layers, 10), 0)
        if number_layers is not None
        else DEFAULT_N_LAYERS
    )
    data["circuit_type"] = circuit_type if circuit_type is not None else DEFAULT_ANSATZ
    data["data_reupload"] = (
        data_reupload if data_reupload is not None else DEFAULT_DATA_REUPLOAD
    )
    data["tffm"] = False  # tffm
    data["seed"] = max(min(seed, 999), 1000) if seed is not None else DEFAULT_SEED

    return data


content = html.Div(
    [
        dash.page_container,
    ],
    className="content",
    id="page-content",
)


app.title = "Favicon"
app.layout = html.Div([sidebar, content])

if __name__ == "__main__":
    args = sys.argv
    if "--debug" in args:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
        )
        logging.info("Running in debug mode")
    else:
        logging.basicConfig(
            level=logging.ERROR, format="%(levelname)s:%(name)s:%(message)s"
        )
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

    logging.info("(Re-)launching Application..")

    try:
        app.run(host="0.0.0.0", port="8050", threaded=True, debug="--debug" in args)
    except Exception as e:
        logging.error(e)
