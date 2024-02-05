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
    ],
    className="sidebar",
    id="page-sidebar",
)

content = html.Div([dash.page_container], className="content", id="page-content")
app.layout = html.Div([sidebar, content])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
