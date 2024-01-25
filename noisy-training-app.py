from dash import (
    Dash,
    dcc,
    html,
    Input,
    State,
    Output,
    callback,
)
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
                html.Div(
                    [
                        dbc.Button(
                            "Training",
                            id="training-button",
                        ),
                        dcc.Loading(
                            id="loading-2",
                            children=[html.Div([html.Div(id="loading-output-2")])],
                            type="circle",
                        ),
                        dcc.Interval(
                            id="interval-component",
                            interval=2 * 1000,  # in milliseconds
                            n_intervals=0,
                        ),
                    ],
                ),
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
                dcc.Graph(id="fig-hist", style={"display": "inline-block"}),
                dcc.Graph(id="fig-expval", style={"display": "inline-block"}),
                dcc.Graph(id="fig-metric"),
            ],
            style={
                "margin-left": "100px",
                "margin-right": "100px",
            },
        ),
    ]
)


instructor = Instructor(2, 6)


@callback(
    Output("fig-hist", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_hist(n):
    # if len(instructor.y) == 0:
    #     return None

    fig_hist = go.Figure(
        data=[
            go.Surface(
                **instructor.get_hist(),
            )
        ]
    )
    fig_hist.update_layout(
        title="Histogram (Absolute Value)",
        template="simple_white",
        width=500,
        height=500,
        # margin=dict(l=65, r=50, b=65, t=90),
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
    )
    # fig_hist.update_layout(
    #     template="simple_white",
    # )

    # instructor.weights = instructor.step(
    #     instructor.weights, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
    # )
    return fig_hist


@callback(
    Output("fig-expval", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_expval(n):
    fig_expval = go.Figure()
    # instructor.y = rng.random(size=len(instructor.y_d))

    if len(instructor.pred) > 0:
        fig_expval.add_scatter(x=instructor.x_d, y=instructor.pred)
    fig_expval.add_scatter(x=instructor.x_d, y=instructor.y_d)

    fig_expval.update_layout(
        title="Prediction",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="Expectation Value",
        yaxis_range=[-1, 1],
        autosize=False,
        width=1800,
        height=500,
    )

    return fig_expval


@callback(
    Output("fig-metric", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_expval(n):
    fig_expval = go.Figure()

    if len(instructor.loss) > 0:
        fig_expval.add_scatter(y=instructor.loss)

    fig_expval.update_layout(
        title="Loss",
        template="simple_white",
        xaxis_title="Step",
        yaxis_title="Loss",
        xaxis_range=[0, instructor.steps],
        autosize=False,
        width=2400,
        height=400,
    )

    return fig_expval


@callback(
    Output("loading-output-2", "children"),
    Input("training-button", "n_clicks"),
    State("bit-flip-prob", "value"),
    State("phase-flip-prob", "value"),
    State("amplitude-damping-prob", "value"),
    State("phase-damping-prob", "value"),
    State("depolarization-prob", "value"),
)
def training(n, bf, pf, ad, pd, dp):
    if n is None:
        return None
    _weights = instructor.weights.copy()
    for s in range(instructor.steps):
        pred, _weights, cost = instructor.step(
            _weights, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
        )

        coeffs = coefficients(
            partial(
                instructor.forward, weights=_weights, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
            ),
            1,
            instructor.max_freq,
        )
        nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
        data_len = len(data["real"][0])
        data["comb"] = np.sqrt(data["real"] ** 2 + data["imag"] ** 2)

        instructor.x = np.arange(-data_len // 2 + 1, data_len // 2 + 1, 1)
        instructor.pred = pred
        instructor.loss.append(cost)
        instructor.append_hist(data["comb"][0])
    return instructor.cost(_weights, instructor.y_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)


if __name__ == "__main__":
    app.run(debug=True)
