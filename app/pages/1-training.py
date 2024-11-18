import dash
import numpy as np
from dash import (
    Input,
    State,
    Output,
    callback,
)
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from typing import Dict, Any, List, Optional

from utils.instructor import Instructor

import logging

log = logging.getLogger(__name__)

dash.register_page(__name__, name="Training")

from layouts.training_page_layout import layout  # noqa: E402
from layouts.training_page_layout import (
    DEFAULT_N_STEPS,
    DEFAULT_N_FREQS,
    DEFAULT_STEPSIZE,
)
from layouts.app_page_layout import (
    DEFAULT_N_QUBITS,
    DEFAULT_N_LAYERS,
    DEFAULT_SEED,
    DEFAULT_DATA_REUPLOAD,
    DEFAULT_ANSATZ,
)


instructor = Instructor(
    DEFAULT_N_QUBITS,
    DEFAULT_N_LAYERS,
    n_freqs=DEFAULT_N_FREQS,
    stepsize=DEFAULT_STEPSIZE,
    seed=DEFAULT_SEED,
    circuit_type=DEFAULT_ANSATZ,
    data_reupload=DEFAULT_DATA_REUPLOAD,
)


def reset_log() -> Dict[str, list]:
    """
    Resets the training log to contain empty lists for the following keys:
    - loss
    - y_hat
    - params
    - ent_cap
    - x
    - y
    - X
    - Y
    - steps

    Returns:
        A dictionary with the given keys and empty lists as values.
    """
    return {
        "loss": [],
        "y_hat": [],
        "params": [],
        "ent_cap": [],
        "ctrl_params": [],
        "x": [],
        "y": [],
        "X": [],
        "Y": [],
        "steps": [],
    }


@callback(
    [
        Output("training-page-storage", "data"),
        Output("training-log-storage", "data", allow_duplicate=True),
        Output("training-start-button", "children", allow_duplicate=True),
    ],
    [
        Input("main-storage", "modified_timestamp"),
        Input("training-bit-flip-prob-slider", "value"),
        Input("training-phase-flip-prob-slider", "value"),
        Input("training-amplitude-damping-prob-slider", "value"),
        Input("training-phase-damping-prob-slider", "value"),
        Input("training-depolarization-prob-slider", "value"),
        Input("training-freqs-numeric-input", "value"),
        Input("training-steps-numeric-input", "value"),
        Input("training-stepsize-numeric-input", "value"),
        Input("training-start-button", "n_clicks"),
    ],
    State("training-start-button", "children"),
    State("main-storage", "data"),
    State("training-page-storage", "data"),
    prevent_initial_call="initial_duplicate",
)
def on_preference_changed(
    _: int,
    bf: float,
    pf: float,
    ad: float,
    pd: float,
    dp: float,
    n_freqs: int,
    steps: int,
    stepsize: int,
    n: int,
    state: str,
    main_data: Dict,
    page_data: Dict,
) -> list:
    """
    Handles the preference change events and updates the training configuration.

    Args:
        _: Unused, represents the modified timestamp of the main storage.
        bf: Bit flip probability from the slider input.
        pf: Phase flip probability from the slider input.
        ad: Amplitude damping probability from the slider input.
        pd: Phase damping probability from the slider input.
        dp: Depolarization probability from the slider input.
        n_freqs: Number of frequencies from the numeric input.
        steps: Number of training steps from the numeric input.
        n: Number of clicks on the start button.
        state: The current text on the training start button.

    Returns:
        A list containing:
            - Updated page data dictionary.
            - Reset log dictionary.
            - Button text indicating the next state.
    """
    page_data = {
        "noise_params": {
            "BitFlip": bf,
            "PhaseFlip": pf,
            "AmplitudeDamping": ad,
            "PhaseDamping": pd,
            "Depolarization": dp,
        },
        "steps": steps if steps is not None and steps > 0 else DEFAULT_N_STEPS,
        "n_freqs": n_freqs if n_freqs is not None and n_freqs > 0 else DEFAULT_N_FREQS,
        "stepsize": (
            stepsize if stepsize is not None and stepsize > 0 else DEFAULT_STEPSIZE
        ),
    } | page_data

    page_log_training = reset_log()

    global instructor

    instructor = Instructor(
        main_data["number_qubits"],
        main_data["number_layers"],
        n_freqs=page_data["n_freqs"],
        stepsize=page_data["stepsize"],
        seed=main_data["seed"],
        circuit_type=main_data["circuit_type"],
        data_reupload=main_data["data_reupload"],
    )

    if state == "Reset Training" or n is None or page_data["lastn"] == n:
        page_data["lastn"] = n
        page_data["running"] = False
        return [page_data, page_log_training, "Start Training"]
    else:
        page_data["lastn"] = n
        page_data["running"] = True
        return [page_data, page_log_training, "Reset Training"]


@callback(
    Output("training-metric-figure", "figure"),  # type: ignore
    Input("training-log-storage", "modified_timestamp"),
    [
        State("training-log-storage", "data"),  # type: ignore
        State("training-page-storage", "data"),  # type: ignore
    ],
    prevent_initial_call=True,
)
def update_loss(
    n: int,  # modified_timestamp
    page_log_training: Dict[str, List[float]],  # type: ignore
    page_data: Dict[str, Any],  # type: ignore
) -> go.Figure:
    """
    Update the "Loss" figure with the latest loss values stored in the
    training-log-storage.

    Args:
        n: The number of times the training-log-storage has been modified.
        page_log_training: The log of the training process.
        page_data: The current page data.

    Returns:
        The updated figure.
    """
    fig_expval = go.Figure()
    if (
        page_log_training is not None
        and len(page_log_training["loss"]) > 0
        and page_data is not None
    ):
        fig_expval.add_scatter(y=page_log_training["loss"])

    fig_expval.update_layout(
        title="Loss",
        template="simple_white",
        xaxis_title="Step",
        yaxis_title="Loss",
        xaxis_range=[
            0,
            page_data["steps"] if page_data is not None else DEFAULT_N_STEPS,
        ],
        autosize=False,
    )

    return fig_expval


@callback(
    Output("training-log-storage", "data", allow_duplicate=True),
    Input("training-log-storage", "modified_timestamp"),
    [
        State("training-log-storage", "data"),
        State("training-page-storage", "data"),
    ],
    prevent_initial_call=True,
)
def pong(
    modified_timestamp: int,
    page_log_training: Dict[str, Any],
    page_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    This callback ensures that the training log is updated continuously.

    The function takes the modified timestamp of the training log and the current data
    of the page log and page data as inputs.
    It returns the current data of the page log.

    If the page log is None, the page data is None, the length of the loss in page log
    is greater than the steps in page data, or the running flag in page data is False,
    the function raises a PreventUpdate exception.

    Otherwise, the function returns the current page log.
    """
    if (
        page_log_training is None
        or page_data is None
        or len(page_log_training["loss"]) > page_data["steps"]
        or not page_data["running"]
    ):
        raise PreventUpdate()
    return page_log_training


@callback(
    Output("training-hist-fig", "figure"),
    Input("training-log-storage", "modified_timestamp"),
    State("training-log-storage", "data"),  # type: Dict[str, Any]
    State("training-page-storage", "data"),
    prevent_initial_call=True,
)
def update_hist(
    n: int,  # modified_timestamp
    page_log_training: Dict[str, Any],  # page_log_storage
    page_data: Dict[str, Any],
) -> go.Figure:  # return type
    """
    Updates the histogram plot with the latest data from the training log.

    :param n: modified_timestamp from the dcc.Store
    :param page_log_training: data from the dcc.Store
    :return: a go.Figure with the latest data
    """
    fig_hist = go.Figure()

    if (
        page_log_training is not None
        and len(page_log_training["loss"]) > 0
        and page_data is not None
    ):
        fig_hist.add_surface(
            x=np.array(page_log_training["X"]),
            y=np.array(page_log_training["steps"]),
            z=np.array(page_log_training["Y"]),
            showscale=False,
            showlegend=False,
        )

    fig_hist.update_layout(
        title="Histogram (Absolute Value)",
        template="simple_white",
        scene=dict(
            xaxis=dict(
                title="Frequency",
            ),
            yaxis=dict(title="Step"),
            zaxis=dict(
                title="Amplitude",
            ),
        ),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1.2),
            center=dict(x=0.1, y=0, z=-0.2),
            eye=dict(x=0.95, y=1.85, z=0.75),
        ),
        coloraxis_showscale=False,
    )

    return fig_hist


@callback(
    Output("training-expval-figure", "figure"),  # type: ignore
    Input("training-log-storage", "modified_timestamp"),
    State("training-log-storage", "data"),  # type: Dict[str, Any]
    State("training-page-storage", "data"),
    prevent_initial_call=True,
)
def update_expval(
    n: int,  # modified_timestamp
    page_log_training: Dict[str, Any],  # page_log_storage
    page_data: Dict[str, Any],
) -> go.Figure:  # return type
    """
    Updates the expectation value plot with the latest data from the training log.

    :param n: modified_timestamp from the dcc.Store
    :param page_log_training: data from the dcc.Store
    :return: a go.Figure with the latest data
    """
    fig_expval = go.Figure()

    if (
        page_log_training is not None
        and len(page_log_training["loss"]) > 0
        and page_data is not None
    ):
        fig_expval.add_scatter(
            x=page_log_training["x"], y=page_log_training["y_hat"], name="Prediction"
        )
        fig_expval.add_scatter(
            x=page_log_training["x"], y=page_log_training["y"], name="Target"
        )

    miny = np.min(page_log_training["y"]) if len(page_log_training["y"]) > 0 else -1
    maxy = np.max(page_log_training["y"]) if len(page_log_training["y"]) > 0 else 1

    fig_expval.update_layout(
        title="Output",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="Expectation Value",
        yaxis_range=[miny, maxy],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig_expval


@callback(
    Output("fig-training-ent", "figure"),
    Input("training-log-storage", "modified_timestamp"),
    [
        State("training-log-storage", "data"),
        State("training-page-storage", "data"),
    ],
    prevent_initial_call=True,
)
def update_ent_cap(
    n: int,  # The modified timestamp of the training log storage
    page_log_training: Dict[str, Any],  # The data in the training log storage
    page_data: Dict[str, Any],  # The data in the training page storage
) -> go.Figure:  # The updated figure
    """
    This function is called when the data in the training log storage changes.
    It creates a line plot of the entangling capability over the training steps.

    Args:
        n: The modified timestamp of the training log storage.
        page_log_training: The current data in the training log storage.
        page_data: The current data in the training page storage.

    Returns:
        The updated figure for the entangling capability.
    """
    fig_ent_cap = go.Figure()
    if (
        page_log_training is not None
        and len(page_log_training["ent_cap"]) > 0
        and page_data is not None
    ):
        fig_ent_cap.add_scatter(
            name="Entangling Capability", y=page_log_training["ent_cap"]
        )

        if len(page_log_training["ctrl_params"]) > 0:
            fig_ent_cap.add_scatter(
                name="Control Parameters", y=page_log_training["ctrl_params"]
            )

    fig_ent_cap.update_layout(
        title="Entangling Capability",
        template="simple_white",
        xaxis_title="Step",
        yaxis_title="Entangling Capability",
        xaxis_range=[
            0,
            page_data["steps"] if page_data is not None else DEFAULT_N_STEPS,
        ],
        autosize=False,
    )

    return fig_ent_cap


@callback(
    Output("training-log-storage", "data"),
    Input("training-log-storage", "data"),
    [
        State("training-page-storage", "data"),
        State("main-storage", "data"),
    ],
    prevent_initial_call=True,
)
def training(
    page_log_training: Dict[str, Any],
    page_data: Dict[str, Any],
    main_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    This function is called when the data in the training log storage changes.
    It runs the training step for the current parameters and noise levels,
    and updates the training log storage with the new parameters, cost, and
    prediction.

    Args:
        page_log_training: The current data in the training log storage.
        page_data: The current data in the training page storage.
        main_data: The current data in the main storage.

    Returns:
        The updated data in the training log storage.
    """
    if page_log_training is None or page_data is None or page_data["running"] is False:
        raise PreventUpdate()

    if len(page_log_training["loss"]) > page_data["steps"]:
        page_log_training = reset_log()

    page_log_training["x"] = instructor.x_d
    page_log_training["y"] = instructor.y_d

    try:
        data = instructor.calc_hist(
            params=instructor.model.params,
            noise_params=page_data["noise_params"],
        )
        page_log_training["X"] = np.arange(
            -len(data) // 2 + 1, len(data) // 2 + 1, 1
        ).tolist()

        page_log_training["Y"].append(data.tolist())

        if main_data["number_qubits"] > 1:
            instructor.model.params = instructor.model.params
            ent_cap = instructor.meyer_wallach(
                noise_params=page_data["noise_params"],
            )

            page_log_training["ent_cap"].append(ent_cap)

            control_params = np.array(
                [
                    instructor.model.pqc.get_control_angles(
                        params, instructor.model.n_qubits
                    )
                    for params in instructor.model.params
                ]
            )
            if control_params.any() != None:
                control_rotation_mean = np.sum(np.abs(control_params) % (2 * np.pi)) / (
                    control_params.size * (2 * np.pi)
                )

                page_log_training["ctrl_params"].append(control_rotation_mean)
            else:
                page_log_training["ctrl_params"] = []

    except Exception as e:
        log.error(e)

    cost, pred = instructor.step(page_data["noise_params"])

    page_log_training["loss"].append(cost.item())
    page_log_training["steps"] = [i for i in range(len(page_log_training["loss"]))]
    page_log_training["y_hat"] = pred

    return page_log_training
