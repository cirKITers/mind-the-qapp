import dash
from dash import Input, Output, dcc, html, callback, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

import pennylane.numpy as np
from pennylane.fourier import coefficients
from pennylane.fourier.visualize import _extract_data_and_labels
from functools import partial

from utils.instructor import Instructor

dash.register_page(__name__, name="Noise Viz")

layout = html.Div(
    [
        html.Div(
            [
                dcc.Store(id="storage-noise-viz", storage_type="session"),
                html.Div(
                    [
                        dbc.Label("Bit-Flip Probability"),
                        dcc.Slider(0, 1, 0.1, value=0, id="bit-flip-prob"),
                        dbc.Label("Phase Flip Probability"),
                        dcc.Slider(0, 1, 0.1, value=0, id="phase-flip-prob"),
                        dbc.Label(
                            "Amplitude Damping Probability",
                        ),
                        dcc.Slider(0, 1, 0.1, value=0, id="amplitude-damping-prob"),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dbc.Label("Phase Damping Probability"),
                        dcc.Slider(0, 1, 0.1, value=0, id="phase-damping-prob"),
                        dbc.Label("Depolarization Probability"),
                        dcc.Slider(0, 1, 0.1, value=0, id="depolarization-prob"),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
            ],
            id="input-container",
        ),
        html.Div(
            [
                dcc.Graph(id="fig-hist"),
                dcc.Graph(id="fig-expval"),
            ],
            id="output-container",
        ),
    ]
)


instructor = Instructor(2, 4)


@callback(
    Output("storage-noise-viz", "data"),
    [
        Input("bit-flip-prob", "value"),
        Input("phase-flip-prob", "value"),
        Input("amplitude-damping-prob", "value"),
        Input("phase-damping-prob", "value"),
        Input("depolarization-prob", "value"),
    ],
    State("storage-noise-viz", "data"),
)
def on_preference_changed(bf, pf, ad, pd, dp, data):
    if bf is None or pf is None or ad is None or pd is None or dp is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    # Give a default data dict with 0 clicks if there's no data.
    data = dict(bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

    return data


@callback(
    [
        Output("fig-hist", "figure"),
        Output("fig-expval", "figure"),
        Output("loading-state", "children"),
    ],
    [
        Input("storage-main", "data"),
        Input("storage-noise-viz", "data"),
    ],
)
def update_output(main_data, page_data):
    bf, pf, ad, pd, dp = (
        page_data["bf"],
        page_data["pf"],
        page_data["ad"],
        page_data["pd"],
        page_data["dp"],
    )
    instructor = Instructor(main_data["niq"], main_data["nil"])
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
    return [fig_hist, fig_expval, "Ready"]
