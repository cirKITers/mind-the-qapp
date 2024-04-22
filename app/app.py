import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, callback, State
from utils.ansaetze import Ansaetze
from typing import Any, Dict, Optional

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], use_pages=True
)

sidebar = html.Div(
    [
        dcc.Store(id="storage-main", storage_type="session"),
        html.Div(
            [
                html.H2(
                    f"Noise App",
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
                            value=1,
                            id="numeric-input-qubits",
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
                            value=1,
                            id="numeric-input-layers",
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
                            placeholder="No Ansatz",
                            required=True,
                            id="select-input-circuit-ident",
                        ),
                    ],
                    className="numeric-input",
                ),
                html.Div(
                    [
                        dbc.Label("Data-Reupload"),
                        dbc.Switch(
                            id="switch-data-reupload", value=True, className="fs-4"
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
                            value=100,
                            id="numeric-input-seed",
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
                    [html.H6("", id="loading-state")],
                    color="primary",
                    type="grow",
                    id="loading-spinner",
                )
            ],
            className="spinnerBox",
        ),
    ],
    className="sidebar",
    id="page-sidebar",
)


@callback(
    Output("storage-main", "data"),
    [
        Input("numeric-input-qubits", "value"),
        Input("numeric-input-layers", "value"),
        Input("select-input-circuit-ident", "value"),
        Input("switch-data-reupload", "value"),
        # Input("switch-tffm", "value"),
        Input("numeric-input-seed", "value"),
    ],
    State("storage-main", "data"),
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
    data["number_qubits"] = max(min(number_qubits, 10), 0)
    data["number_layers"] = max(min(number_layers, 10), 0)
    data["circuit_type"] = circuit_type
    data["data_reupload"] = data_reupload
    data["tffm"] = False  # tffm
    data["seed"] = max(min(seed, 999), 100)

    return data


content = html.Div(
    [
        dash.page_container,
    ],
    className="content",
    id="page-content",
)


app.layout = html.Div([sidebar, content])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
