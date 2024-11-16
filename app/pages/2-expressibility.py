import dash
import numpy as np
from dash import (
    Input,
    State,
    Output,
    callback,
)
import plotly.graph_objects as go

from typing import Dict

from utils.instructor import Instructor
from utils.validation import data_is_valid

dash.register_page(__name__, name="Expressibility")

from layouts.expressibility_page_layout import layout  # noqa: E402


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
    Output("expr-kl-noise-figure", "figure"),
    [
        Input("expr-page-storage", "data"),
    ],
    State("main-storage", "data"),
    prevent_initial_call=True,
)
def update_kl_noise(page_data, main_data):
    fig_kl = go.Figure()
    fig_kl.update_layout(
        title="KL Divergence over Noise",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="KL Divergence",
    )

    if not data_is_valid(page_data, main_data):
        return fig_kl

    n_samples, n_input_samples, n_bins = (
        page_data["n_samples"],
        page_data["n_input_samples"],
        page_data["n_bins"],
    )

    if main_data["circuit_type"] is None:
        return fig_kl

    class NoiseDict(Dict[str, float]):
        """
        A dictionary subclass for noise params.
        """

        def __truediv__(self, other: float) -> "NoiseDict":
            """
            Divide all values by a scalar.
            """
            return NoiseDict({k: v / other for k, v in self.items()})

        def __mul__(self, other: float) -> "NoiseDict":
            """
            Multiply all values by a scalar.
            """
            return NoiseDict({k: v * other for k, v in self.items()})

    noise_params = NoiseDict(page_data["noise_params"])

    instructor = Instructor(
        main_data["number_qubits"],
        main_data["number_layers"],
        seed=main_data["seed"],
        circuit_type=main_data["circuit_type"],
        data_reupload=main_data["data_reupload"],
    )
    _, y_haar = instructor.haar_integral(n_bins)

    kl_divergence = []
    noise_steps = 5
    for step in range(noise_steps + 1):  # +1 to go for 100%
        part_noise_params = noise_params * (step / noise_steps)

        # sample state fidelities for increasing noise

        _, _, fidelity_score = instructor.state_fidelities(
            n_samples=n_samples,
            n_bins=n_bins,
            n_input_samples=0,
            noise_params=part_noise_params,
        )

        kl_divergence.append(instructor.kullback_leibler(fidelity_score, y_haar).item())

    fig_kl.add_scatter(x=list(range(noise_steps + 1)), y=kl_divergence)
    fig_kl.update_layout(
        yaxis_range=[0, max(kl_divergence) + 0.2],
        xaxis_title="Noise",
        yaxis_title="KL Divergence",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(noise_steps + 1)),
            ticktext=[f"{i / noise_steps * 100:.0f}%" for i in range(noise_steps + 1)],
        ),
    )

    return fig_kl


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
        title="KL Divergence over Inputs",
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
        page_data["n_input_samples"] if page_data["n_input_samples"] > 1 else 0,
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

    if n_input_samples == 0:
        fig_expr.add_scatter(x=fidelity_values, y=fidelity_score)
        fig_expr.update_layout(
            xaxis_title="Fidelity",
            yaxis_title="Prob. Density",
        )
    else:
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
