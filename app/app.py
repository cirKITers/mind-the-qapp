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
    Output("session", "data"),
    [
        Input("numeric-input-qubits", "value"),
        Input("numeric-input-layers", "value"),
    ],
    State("session", "data"),
)
def on_preference_changed(niq, nil, data):
    if niq is None and nil is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    # Give a default data dict with 0 clicks if there's no data.
    data = data or {}
    data["niq"] = max(min(niq, 10), 0)
    data["nil"] = max(min(nil, 10), 0)

    return data


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
