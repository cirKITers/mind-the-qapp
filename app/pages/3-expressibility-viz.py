import dash
import numpy as np
from dash import (
    Dash,
    dcc,
    html,
    Input,
    State,
    Output,
    callback,
)
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Fourier coefficients utils
from pennylane.fourier import coefficients
from functools import partial
from pennylane.fourier.visualize import _extract_data_and_labels

from utils.instructor import Instructor
from utils.expressibility import (
    Expressibility_Sampler,
    get_sampled_haar_probability_histogram,
)

dash.register_page(__name__, name="Expr. Viz")

layout = html.Div(
    [
        dcc.Store(id="storage-expr-viz", storage_type="session"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [dbc.Label("Circuit Type")],
                            style={
                                "width": "12vh",
                                "display": "inline-block",
                                "padding": "0 10px",
                            },
                        ),
                        html.Div(
                            [
                                dbc.Input(
                                    type="number",
                                    # currently only circ 19 is supported
                                    min=19,
                                    max=19,
                                    step=1,
                                    value=19,
                                    id="circuit-type",
                                ),
                            ],
                            style={"width": "10vh", "display": "inline-block"},
                        ),
                    ],
                    style={"width": "100%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Div(
                            [dbc.Label("Sampled Parameter Pairs:")],
                            style={
                                "width": "12vh",
                                "display": "inline-block",
                                "padding": "0 10px",
                            },
                        ),
                        html.Div(
                            [
                                dbc.Input(
                                    type="number",
                                    min=1,
                                    max=5000,
                                    step=1,
                                    value=1000,
                                    id="num-param-sample-pairs",
                                ),
                            ],
                            style={"width": "10vh", "display": "inline-block"},
                        ),
                    ],
                    style={"width": "100%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Div(
                            [dbc.Label("Input Samples")],
                            style={
                                "width": "12vh",
                                "display": "inline-block",
                                "padding": "0 10px",
                            },
                        ),
                        html.Div(
                            [
                                dbc.Input(
                                    type="number",
                                    min=2,
                                    max=100,
                                    step=1,
                                    value=5,
                                    id="num-input-samples",
                                ),
                            ],
                            style={"width": "10vh", "display": "inline-block"},
                        ),
                    ],
                    style={"width": "100%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Div(
                            [dbc.Label("Histogram Bins")],
                            style={
                                "width": "12vh",
                                "display": "inline-block",
                                "padding": "0 10px",
                            },
                        ),
                        html.Div(
                            [
                                dbc.Input(
                                    type="number",
                                    min=1,
                                    max=500,
                                    step=1,
                                    value=75,
                                    id="num-histogram-bins",
                                ),
                            ],
                            style={"width": "10vh", "display": "inline-block"},
                        ),
                    ],
                    style={"width": "100%", "display": "inline-block"},
                ),
            ],
        ),
        html.Div(
            [
                dcc.Graph(
                    id="fig-hist-haar",
                    style={
                        "display": "inline-block",
                        "height": "100%",
                        "width": "49%",
                    },
                ),
                dcc.Graph(
                    id="fig-hist-expr",
                    style={
                        "display": "inline-block",
                        "height": "100%",
                        "width": "49%",
                    },
                ),
            ],
            style={"height": "49%", "width": "100%", "display": "inline-block"},
        ),
        html.Div(
            [
                dcc.Graph(
                    id="fig-hist-fourier",
                    style={
                        "display": "inline-block",
                        "height": "100%",
                        "width": "49%",
                    },
                ),
                html.Div(
                    dbc.Spinner(
                        color="primary",
                        type="grow",
                        id="loading-spinner-expr",
                    ),
                    style={
                        "display": "inline-block",
                        "height": "100%",
                        "width": "49%",
                    },
                ),
            ],
            style={"height": "49%", "width": "100%", "display": "inline-block"},
        ),
    ]
)


@callback(
    Output("storage-expr-viz", "data"),
    Output("loading-spinner-expr", "children", allow_duplicate=True),
    [
        Input("circuit-type", "value"),
        Input("num-param-sample-pairs", "value"),
        Input("num-input-samples", "value"),
        Input("num-histogram-bins", "value"),
    ],
    prevent_initial_call=True,
)
def on_preference_changed(circ_type, n_samples, n_input_samples, n_bins):

    # Give a default data dict with 0 clicks if there's no data.
    data = dict(
        circ_type=circ_type,
        n_samples=n_samples,
        n_input_samples=n_input_samples,
        n_bins=n_bins,
    )

    return data, "Loaded data"


@callback(
    [
        Output("fig-hist-fourier", "figure"),
        Output("fig-hist-haar", "figure"),
    ],
    [
        Input("storage-main", "data"),
        Input("storage-expr-viz", "modified_timestamp"),
    ],
    State("storage-expr-viz", "data"),
    prevent_initial_call=True,
)
def update_output(main_data, _, page_data):
    if page_data is None or main_data is None:
        return [go.Figure(), go.Figure(), "Not Ready"]
    circ_type, n_samples, n_input_samples, n_bins = (
        page_data["circ_type"],
        page_data["n_samples"],
        page_data["n_input_samples"],
        page_data["n_bins"],
    )
    instructor = Instructor(
        main_data["number_qubits"], main_data["number_layers"], seed=main_data["seed"]
    )
    coeffs = coefficients(
        partial(instructor.forward),
        1,
        instructor.max_freq,
    )
    nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
    data_len = len(data["real"][0])
    data["comb"] = np.sqrt(data["real"] ** 2 + data["imag"] ** 2)

    fig_coeffs = go.Figure()
    fig_coeffs.add_bar(
        x=np.arange(-data_len // 2 + 1, data_len // 2 + 1, 1), y=data["comb"][0]
    )
    fig_coeffs.update_layout(
        title="Histogram (Absolute Value)",
        template="simple_white",
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
        yaxis_range=[0, 0.5],
    )

    x_haar, y_haar = get_sampled_haar_probability_histogram(
        main_data["number_qubits"], n_bins, n_input_samples
    )

    fig_haar = go.Figure()
    fig_haar.add_bar(
        x=x_haar,
        y=y_haar,
    )
    fig_haar.update_layout(
        title="Haar Probability Densities",
        template="simple_white",
        xaxis_title="Fidelity",
        yaxis_title="Probability",
        yaxis_range=[0, 0.5],
    )

    return [fig_coeffs, fig_haar]


@callback(
    [
        Output("fig-hist-expr", "figure"),
        Output("loading-spinner-expr", "children", allow_duplicate=True),
    ],
    [
        Input("storage-main", "data"),
        Input("storage-expr-viz", "modified_timestamp"),
    ],
    State("storage-expr-viz", "data"),
    prevent_initial_call=True,
)
def update_output_probabilities(main_data, _, page_data):
    if page_data is None or main_data is None:
        return [go.Figure(), go.Figure(), "Not Ready"]
    circ_type, n_samples, n_input_samples, n_bins = (
        page_data["circ_type"],
        page_data["n_samples"],
        page_data["n_input_samples"],
        page_data["n_bins"],
    )
    expr_sampler = Expressibility_Sampler(
        main_data["number_qubits"],
        main_data["number_layers"],
        main_data["seed"],
        circ_type,
        n_samples,
        n_input_samples,
        n_bins,
    )
    x_samples, y_samples, z_samples = expr_sampler.sample_hist_state_fidelities()

    fig_expr = go.Figure(
        go.Surface(
            x=x_samples,
            y=y_samples,
            z=z_samples,
            cmax=1,
            cmin=0,
        )
    )
    fig_expr.update_layout(
        title="Probability Densities",
        margin=dict(l=65, r=50, b=65, t=90),
    )

    return [fig_expr, "Ready"]
