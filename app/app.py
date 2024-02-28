import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, callback, State
from dash.exceptions import PreventUpdate

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], use_pages=True
)


sidebar = html.Div(
    [
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
                dcc.Store(id="storage-main", storage_type="session"),
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
                        dbc.Label("Circuit Identifier"),
                        dbc.Input(
                            type="number",
                            min=19,
                            max=19,
                            step=1,
                            value=19,
                            id="numeric-input-circuit-ident",
                        ),
                    ],
                    className="numeric-input",
                ),
                html.Div(
                    [
                        dbc.Label("Data-Reupload"),
                        dbc.Switch(id="switch-data-reupload", value=True),
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

content = html.Div([dash.page_container], className="content", id="page-content")
app.layout = html.Div([sidebar, content])


@callback(
    Output("storage-main", "data"),
    [
        Input("numeric-input-qubits", "value"),
        Input("numeric-input-layers", "value"),
        Input("numeric-input-circuit-ident", "value"),
        Input("switch-data-reupload", "value"),
        # Input("switch-tffm", "value"),
        Input("numeric-input-seed", "value"),
    ],
    State("storage-main", "data"),
    # prevent_initial_call=True,
)
def on_preference_changed(
    number_qubits, number_layers, circuit_type, data_reupload, seed, data
):
    # Give a default data dict with 0 clicks if there's no data.
    data = data or {}
    data["number_qubits"] = max(min(number_qubits, 10), 0)
    data["number_layers"] = max(min(number_layers, 10), 0)
    data["circuit_type"] = circuit_type
    data["data_reupload"] = data_reupload
    data["tffm"] = False  # tffm
    data["seed"] = max(min(seed, 999), 100)

    return data


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
