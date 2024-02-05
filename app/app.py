import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, callback

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
                html.Div(
                    [
                        dbc.Label("# of Qubits (0-10)"),
                        dbc.Input(
                            type="number",
                            min=0,
                            max=10,
                            step=1,
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
