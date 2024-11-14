import dash
import numpy as np
from dash import (
    Input,
    State,
    Output,
    callback,
)
import plotly.graph_objects as go

from utils.instructor import Instructor
from utils.validation import data_is_valid

from layouts.expressibility_page_layout import layout  # noqa

dash.register_page(__name__, name="Expressibility")


@callback(
    Output("expr-page-storage", "data", allow_duplicate=True),
    Input("main-storage", "modified_timestamp"),
    State("expr-page-storage", "data"),
    prevent_initial_call=True,
)
def update_page_data(_, page_data):
    return page_data


@callback(
    Output("expr-page-storage", "data"),
    [
        Input("expr-param-sample-pairs-input", "value"),
        Input("expr-samples-input", "value"),
        Input("expr-histogram-bins-input", "value"),
        Input("expr-bit-flip-prob-slider", "value"),
        Input("expr-phase-flip-prob-slider", "value"),
        Input("expr-amplitude-damping-prob-slider", "value"),
        Input("expr-phase-damping-prob-slider", "value"),
        Input("expr-depolarization-prob-slider", "value"),
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
    data = {
        "n_samples": n_samples,
        "n_input_samples": n_input_samples,
        "n_bins": n_bins,
        "noise_params": {
            "BitFlip": bf,
            "PhaseFlip": pf,
            "AmplitudeDamping": ad,
            "PhaseDamping": pd,
            "Depolarization": dp,
        },
    }

    return data


@callback(
    [
        Output("fig-hist-fourier", "figure"),
    ],
    [
        Input("expr-page-storage", "data"),
    ],
    State("main-storage", "data"),
    prevent_initial_call=True,
)
def update_hist_fourier(page_data, main_data):
    fig_coeffs = go.Figure()
    fig_coeffs.update_layout(
        title="Histogram (Absolute Value)",
        template="simple_white",
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
    )

    if not data_is_valid(page_data, main_data):
        return [fig_coeffs]

    instructor = Instructor(
        main_data["number_qubits"],
        main_data["number_layers"],
        seed=main_data["seed"],
        circuit_type=main_data["circuit_type"],
        data_reupload=main_data["data_reupload"],
    )

    data = instructor.calc_hist(
        instructor.model.params, noise_params=page_data["noise_params"]
    )

    data_len = len(data)

    fig_coeffs.add_bar(x=np.arange(-data_len // 2 + 1, data_len // 2 + 1, 1), y=data)

    return [fig_coeffs]


@callback(
    [
        Output("expr-hist-figure", "figure"),
        Output("expr-kl-figure", "figure"),
        Output("expr-haar-figure", "figure"),
        Output("main-loading-state", "children", allow_duplicate=True),
    ],
    [
        Input("expr-page-storage", "data"),
    ],
    State("main-storage", "data"),
    prevent_initial_call=True,
)
def update_output_probabilities(page_data, main_data):
    fig_expr = go.Figure()
    fig_expr.update_layout(
        title="Expressibility",
        template="simple_white",
        scene=dict(
            xaxis=dict(
                title="Fidelity",
            ),
            yaxis=dict(title="Input"),
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

    fig_kl = go.Figure()
    fig_kl.update_layout(
        title="KL Divergence",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="KL Divergence",
    )

    fig_haar = go.Figure()
    fig_haar.update_layout(
        title="Haar Probability Densities",
        template="simple_white",
        xaxis_title="Fidelity",
        yaxis_title="Probability",
    )

    if not data_is_valid(page_data, main_data):
        return [fig_expr, fig_kl, "Not Ready"]

    n_samples, n_input_samples, n_bins = (
        page_data["n_samples"],
        page_data["n_input_samples"],
        page_data["n_bins"],
    )

    if main_data["circuit_type"] is None:
        return [fig_expr, "Ready"]

    instructor = Instructor(
        main_data["number_qubits"],
        main_data["number_layers"],
        seed=main_data["seed"],
        circuit_type=main_data["circuit_type"],
        data_reupload=main_data["data_reupload"],
    )

    inputs, fidelity_values, fidelity_score = instructor.state_fidelities(
        n_samples=n_samples,
        n_bins=n_bins,
        n_input_samples=n_input_samples,
        noise_params=page_data["noise_params"],
    )

    fig_expr.add_surface(
        x=fidelity_values,
        y=inputs,
        z=fidelity_score,
        cmax=fidelity_score.max().item(),
        cmin=0,
        showscale=False,
        showlegend=False,
    )

    x_haar, y_haar = instructor.haar_integral(n_bins)

    fig_haar.add_bar(
        x=x_haar,
        y=y_haar,
    )

    kl_divergence = instructor.kullback_leibler(fidelity_score, y_haar)

    fig_kl.add_scatter(x=inputs, y=kl_divergence)
    fig_kl.update_layout(
        yaxis_range=[0, max(kl_divergence) + 0.2],
    )

    return [fig_expr, fig_kl, fig_haar, "Ready"]


@callback(
    Output("expr-ent-cap-badge", "children"),
    [
        Input("expr-page-storage", "data"),
    ],
    State("main-storage", "data"),
    prevent_initial_call=True,
)
def update_ent_cap(page_data, main_data):
    if not data_is_valid(page_data, main_data) or main_data["number_qubits"] == 1:
        return 0

    instructor = Instructor(
        main_data["number_qubits"],
        main_data["number_layers"],
        seed=main_data["seed"],
        circuit_type=main_data["circuit_type"],
        data_reupload=main_data["data_reupload"],
    )

    ent_cap = instructor.meyer_wallach(
        n_samples=10, noise_params=page_data["noise_params"]
    )
    return f"{ent_cap:.3f}"
