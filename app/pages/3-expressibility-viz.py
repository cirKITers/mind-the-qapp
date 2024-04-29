import dash
import numpy as np
from dash import (
    dcc,
    html,
    Input,
    State,
    Output,
    callback,
)
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Fourier coefficients utils
from pennylane.fourier import coefficients
from functools import partial
from pennylane.fourier.visualize import _extract_data_and_labels

from utils.instructor import Instructor
from utils.expressibility import (
    Expressibility_Sampler,
    get_sampled_haar_probability_histogram,
    get_kl_divergence_expr,
)
from utils.validation import data_is_valid

from utils.entangling import EntanglingCapability_Sampler

dash.register_page(__name__, name="Expressibility")

layout = html.Div(
    [
        dcc.Store(id="storage-expr-viz", storage_type="session"),
        html.Div(
            [
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Sampled Parameter Pairs:",
                                                html_for="num-param-sample-pairs",
                                            ),
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                type="number",
                                                min=100,
                                                max=1000,
                                                step=1,
                                                value=200,
                                                id="num-param-sample-pairs",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Input Samples",
                                                html_for="num-input-samples",
                                            ),
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                type="number",
                                                min=2,
                                                max=100,
                                                step=1,
                                                value=5,
                                                id="num-input-samples",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Histogram Bins",
                                                html_for="num-histogram-bins",
                                            ),
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                type="number",
                                                min=10,
                                                max=500,
                                                step=1,
                                                value=20,
                                                id="num-histogram-bins",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Meyer-Wallach Entangling Capability",
                                                html_for="ent-cap",
                                            ),
                                        ),
                                        dbc.Col(
                                            html.H4(
                                                dbc.Badge(
                                                    "0.0",
                                                    color="primary",
                                                    pill=True,
                                                    className="me-1",
                                                    id="ent-cap",
                                                )
                                            ),
                                        ),
                                    ],
                                    style={"padding-top": "10px"},
                                ),
                            ],
                            className="settingsRow",
                        )
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label("Phase Damping Probability"),
                                        ),
                                        dbc.Col(
                                            dcc.Slider(
                                                0,
                                                0.5,
                                                0.05,
                                                value=0,
                                                id="phase-damping-prob-expr",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label("Depolarization Probability"),
                                        ),
                                        dbc.Col(
                                            dcc.Slider(
                                                0,
                                                0.5,
                                                0.05,
                                                value=0,
                                                id="depolarization-prob-expr",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label("Bit-Flip Probability"),
                                        ),
                                        dbc.Col(
                                            dcc.Slider(
                                                0,
                                                0.5,
                                                0.05,
                                                value=0,
                                                id="bit-flip-prob-expr",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label("Phase Flip Probability"),
                                        ),
                                        dbc.Col(
                                            dcc.Slider(
                                                0,
                                                0.5,
                                                0.05,
                                                value=0,
                                                id="phase-flip-prob-expr",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Amplitude Damping Probability",
                                            ),
                                        ),
                                        dbc.Col(
                                            dcc.Slider(
                                                0,
                                                0.5,
                                                0.05,
                                                value=0,
                                                id="amplitude-damping-prob-expr",
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                            className="settingsRow",
                        )
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="fig-hist-expr",
                            style={
                                "display": "inline-block",
                                "height": "50vh",
                                "width": "100%",
                            },
                        ),
                        dcc.Graph(
                            id="fig-expr-kl",
                            style={
                                "display": "inline-block",
                                "height": "30vh",
                                "width": "100%",
                            },
                        ),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="fig-hist-haar",
                            style={
                                "display": "inline-block",
                                "height": "40vh",
                                "width": "100%",
                            },
                        ),
                        dcc.Graph(
                            id="fig-hist-fourier",
                            style={
                                "display": "inline-block",
                                "height": "40vh",
                                "width": "100%",
                            },
                        ),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
            ],
            style={"height": "49%", "display": "inline-block"},
        ),
    ]
)


@callback(
    Output("storage-expr-viz", "data", allow_duplicate=True),
    Input("storage-main", "modified_timestamp"),
    State("storage-expr-viz", "data"),
    prevent_initial_call=True,
)
def update_page_data(_, page_data):
    return page_data


@callback(
    Output("storage-expr-viz", "data"),
    [
        Input("num-param-sample-pairs", "value"),
        Input("num-input-samples", "value"),
        Input("num-histogram-bins", "value"),
        Input("bit-flip-prob-expr", "value"),
        Input("phase-flip-prob-expr", "value"),
        Input("amplitude-damping-prob-expr", "value"),
        Input("phase-damping-prob-expr", "value"),
        Input("depolarization-prob-expr", "value"),
    ],
)
def on_preference_changed(
    n_samples,
    n_input_samples,
    n_bins,
    bf,
    pf,
    ad,
    pd,
    dp,
):

    # Give a default data dict with 0 clicks if there's no data.
    data = dict(
        n_samples=n_samples,
        n_input_samples=n_input_samples,
        n_bins=n_bins,
        bf=bf,
        pf=pf,
        ad=ad,
        pd=pd,
        dp=dp,
    )

    return data


@callback(
    [
        Output("fig-hist-fourier", "figure"),
    ],
    [
        Input("storage-expr-viz", "data"),
    ],
    State("storage-main", "data"),
    prevent_initial_call=True,
)
def update_hist_fourier(page_data, main_data):
    if not data_is_valid(page_data, main_data):
        return [go.Figure()]

    bf, pf, ad, pd, dp = (
        page_data["bf"],
        page_data["pf"],
        page_data["ad"],
        page_data["pd"],
        page_data["dp"],
    )

    instructor = Instructor(
        main_data["number_qubits"],
        main_data["number_layers"],
        seed=main_data["seed"],
        circuit_type=main_data["circuit_type"],
        data_reupload=main_data["data_reupload"],
    )
    coeffs = coefficients(
        partial(instructor.forward, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp),
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
    )

    return [fig_coeffs]


@callback(
    [
        Output("fig-hist-haar", "figure"),
    ],
    [
        Input("storage-expr-viz", "data"),
    ],
    State("storage-main", "data"),
    prevent_initial_call=True,
)
def update_hist_haar(page_data, main_data):
    if not data_is_valid(page_data, main_data):
        return [go.Figure()]

    n_samples, n_input_samples, n_bins = (
        page_data["n_samples"],
        page_data["n_input_samples"],
        page_data["n_bins"],
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
    )

    return [fig_haar]


@callback(
    [
        Output("fig-hist-expr", "figure"),
        Output("fig-expr-kl", "figure"),
        Output("loading-state", "children", allow_duplicate=True),
    ],
    [
        Input("storage-expr-viz", "data"),
    ],
    State("storage-main", "data"),
    prevent_initial_call=True,
)
def update_output_probabilities(page_data, main_data):
    if not data_is_valid(page_data, main_data):
        return [go.Figure(), go.Figure(), "Not Ready"]
    fig_expr = go.Figure()
    fig_expr.update_layout(
        title="Expressibility",
        template="simple_white",
        scene=dict(
            xaxis=dict(
                title="Input",
            ),
            yaxis=dict(title="Fidelity"),
            zaxis=dict(
                title="Prob. Density",
            ),
        ),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0.1, y=0, z=0),
            eye=dict(x=-1.15, y=-2.05, z=1.05),
        ),
        coloraxis_showscale=False,
    )

    n_samples, n_input_samples, n_bins = (
        page_data["n_samples"],
        page_data["n_input_samples"],
        page_data["n_bins"],
    )

    if main_data["circuit_type"] == None:
        return [fig_expr, "Ready"]

    bf, pf, ad, pd, dp = (
        page_data["bf"],
        page_data["pf"],
        page_data["ad"],
        page_data["pd"],
        page_data["dp"],
    )

    expr_sampler = Expressibility_Sampler(
        main_data["number_qubits"],
        main_data["number_layers"],
        main_data["seed"],
        main_data["circuit_type"],
        main_data["data_reupload"],
        n_samples,
        n_input_samples,
        n_bins,
    )
    x_samples, y_samples, z_samples = expr_sampler.sample_hist_state_fidelities(
        bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
    )

    fig_expr.add_surface(
        x=x_samples,
        y=y_samples,
        z=z_samples,
        cmax=1,
        cmin=0,
        showscale=False,
        showlegend=False,
    )

    fig_kl = go.Figure()
    _, y_haar = get_sampled_haar_probability_histogram(
        main_data["number_qubits"], n_bins, n_input_samples
    )

    kl_divergence = get_kl_divergence_expr(np.transpose(z_samples), y_haar)

    fig_kl.update_layout(
        title="KL Divergence",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="KL Divergence",
        yaxis_range=[0, max(kl_divergence) + 0.2],
    )

    fig_kl.add_scatter(x=x_samples, y=kl_divergence)

    return [fig_expr, fig_kl, "Ready"]


@callback(
    Output("ent-cap", "children"),
    [
        Input("storage-expr-viz", "data"),
    ],
    State("storage-main", "data"),
    prevent_initial_call=True,
)
def update_ent_cap(page_data, main_data):
    if not data_is_valid(page_data, main_data) or main_data["number_qubits"] == 1:
        return 0

    bf, pf, ad, pd, dp = (
        page_data["bf"],
        page_data["pf"],
        page_data["ad"],
        page_data["pd"],
        page_data["dp"],
    )

    ent_sampler = EntanglingCapability_Sampler(
        main_data["number_qubits"],
        main_data["number_layers"],
        main_data["seed"],
        main_data["circuit_type"],
        main_data["data_reupload"],
    )
    ent_cap = ent_sampler.calculate_entangling_capability(
        10, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
    )
    return f"{ent_cap:.3f}"
