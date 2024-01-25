from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go

from functools import partial

import pennylane.numpy as np
from pennylane.fourier import coefficients
from pennylane.fourier.visualize import _extract_data_and_labels

from instructor import Instructor

import dash_bootstrap_components as dbc

# from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

# stylesheet with the .dbc class to style  dcc, DataTable and AG Grid components with a Bootstrap theme
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.MATERIA, dbc.icons.FONT_AWESOME, dbc_css],
)


app.layout = html.Div(
    [
        html.Div(
            id="input-container",
            children=[
                html.Label("Bit-Flip Probability", htmlFor="bit-flip-prob"),
                dcc.Slider(0, 1, 0.1, value=0, id="bit-flip-prob"),
                html.Label("Phase Flip Probability", htmlFor="phase-flip-prob"),
                dcc.Slider(0, 1, 0.1, value=0, id="phase-flip-prob"),
                html.Label(
                    "Amplitude Damping Probability", htmlFor="amplitude-damping-prob"
                ),
                dcc.Slider(0, 1, 0.1, value=0, id="amplitude-damping-prob"),
                html.Label("Phase Damping Probability", htmlFor="phase-damping-prob"),
                dcc.Slider(0, 1, 0.1, value=0, id="phase-damping-prob"),
                html.Label("Depolarization Probability", htmlFor="depolarization-prob"),
                dcc.Slider(0, 1, 0.1, value=0, id="depolarization-prob"),
            ],
            style={
                "margin-left": "100px",
                "margin-right": "100px",
                "margin-top": "35px",
                "margin-bottom": "35px",
            },
        ),
        html.Div(
            id="output-container",
            children=[
                dcc.Graph(id="fig-hist"),
                dcc.Graph(id="fig-expval"),
            ],
            style={
                "margin-left": "100px",
                "margin-right": "100px",
            },
        ),
    ]
)


instructor = Instructor(2, 4)


@callback(
    [Output("fig-hist", "figure"), Output("fig-expval", "figure")],
    [
        Input("bit-flip-prob", "value"),
        Input("phase-flip-prob", "value"),
        Input("amplitude-damping-prob", "value"),
        Input("phase-damping-prob", "value"),
        Input("depolarization-prob", "value"),
    ],
)
def update_output(bf, pf, ad, pd, dp):
    coeffs = coefficients(
        partial(instructor.forward, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp),
        1,
        instructor.max_freq,
    )
    nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
    data_len = len(data["real"][0])
    data["comb"] = np.sqrt(data["real"] ** 2 + data["imag"] ** 2)

    y_pred = instructor.forward(
        instructor.x_d, weights=instructor.weights, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
    )

    fig_hist = go.Figure()
    fig_expval = go.Figure()

    fig_expval.add_scatter(x=instructor.x_d, y=y_pred)
    fig_hist.add_bar(
        x=np.arange(-data_len // 2 + 1, data_len // 2 + 1, 1), y=data["comb"][0]
    )

    fig_expval.update_layout(
        title="Prediction",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="Expectation Value",
        yaxis_range=[-1, 1],
    )
    fig_hist.update_layout(
        title="Histogram (Absolute Value)",
        template="simple_white",
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
        yaxis_range=[0, 0.5],
    )
    return [fig_hist, fig_expval]


if __name__ == "__main__":
    app.run(debug=True)
