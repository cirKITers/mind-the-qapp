from layouts.app_page_layout import sidebar_top, sidebar_bottom, content
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html, callback, State
from typing import Any, Dict, Optional
import sys

import logging


app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    use_pages=True,
)

app.title = "Favicon"

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
        max(min(number_qubits, 10), 0) if number_qubits is not None else None
    )
    data["number_layers"] = (
        max(min(number_layers, 10), 0) if number_layers is not None else None
    )
    data["circuit_type"] = circuit_type if circuit_type is not None else "No_Ansatz"
    data["data_reupload"] = data_reupload
    data["tffm"] = False  # tffm
    data["seed"] = max(min(seed, 999), 100)

    return data


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
