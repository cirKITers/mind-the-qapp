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
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from typing import Dict, Any, List, Optional

from utils.instructor import Instructor

import dash_bootstrap_components as dbc

dash.register_page(__name__, name="Training")

DEFAULT_N_STEPS = 10
DEFAULT_N_FREQS = 3

layout = html.Div(
    [
        dcc.Store(id="training-page-storage", storage_type="session"),
        dcc.Store(id="training-log-storage", storage_type="session"),
        html.Div(
            [
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label("Bit-Flip Probability"),
                                        ),
                                        dbc.Col(
                                            dcc.Slider(
                                                min=0,
                                                max=0.1,
                                                step=0.01,
                                                value=0,
                                                id="training-bit-flip-prob-slider",
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
                                                min=0,
                                                max=0.1,
                                                step=0.01,
                                                value=0,
                                                id="training-phase-flip-prob-slider",
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
                                                min=0,
                                                max=0.1,
                                                step=0.01,
                                                value=0,
                                                id="training-amplitude-damping-prob-slider",
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
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.Label("Phase Damping Probability")),
                                        dbc.Col(
                                            dcc.Slider(
                                                min=0,
                                                max=0.1,
                                                step=0.01,
                                                value=0,
                                                id="training-phase-damping-prob-slider",
                                            )
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label("Depolarization Probability")
                                        ),
                                        dbc.Col(
                                            dcc.Slider(
                                                min=0,
                                                max=0.1,
                                                step=0.01,
                                                value=0,
                                                id="training-depolarization-prob-slider",
                                            )
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label("# of Freqs.:"),
                                                    ],
                                                    style={
                                                        "display": "inline-block",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Input(
                                                            type="number",
                                                            min=1,
                                                            max=11,
                                                            step=1,
                                                            value=DEFAULT_N_FREQS,
                                                            id="training-freqs-numeric-input",
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "5vw",
                                                        "display": "inline-block",
                                                        "padding-left": "20px",
                                                    },
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label("Steps:"),
                                                    ],
                                                    style={
                                                        "display": "inline-block",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Input(
                                                            type="number",
                                                            min=1,
                                                            max=101,
                                                            step=1,
                                                            value=DEFAULT_N_STEPS,
                                                            id="training-steps-numeric-input",
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "5vw",
                                                        "display": "inline-block",
                                                        "padding-left": "20px",
                                                    },
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Start Training",
                                                    id="training-start-button",
                                                ),
                                            ],
                                        ),
                                    ],
                                    style={"padding": "15px"},
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
                            id="training-hist-fig",
                            style={
                                "display": "inline-block",
                                "height": "50vh",
                                "width": "100%",
                            },
                        ),
                        dcc.Graph(
                            id="training-metric-figure",
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
                            id="training-expval-figure",
                            style={
                                "display": "inline-block",
                                "height": "40vh",
                                "width": "100%",
                            },
                        ),
                        dcc.Graph(
                            id="fig-training-ent",
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
        Input("training-start-button", "n_clicks"),
    ],
    State("training-start-button", "children"),
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
    n: int,
    state: str,
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
        "steps": steps,
        "n_freqs": n_freqs,
        "running": state != "Reset Training",
    }
    page_log_training = reset_log()

    if state == "Reset Training":
        return [page_data, page_log_training, "Start Training"]
    else:
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
    page_log_training: Optional[Dict[str, Any]],
    page_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    This callback ensures that the training log is updated continuously.

    The function takes the modified timestamp of the training log and the current data
    of the page log and page data as inputs. It returns the current data of the page log.

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
    prevent_initial_call=True,
)
def update_hist(
    n: int,  # modified_timestamp
    page_log_training: Optional[Dict[str, Any]],  # page_log_storage
) -> go.Figure:  # return type
    """
    Updates the histogram plot with the latest data from the training log.

    :param n: modified_timestamp from the dcc.Store
    :param page_log_training: data from the dcc.Store
    :return: a go.Figure with the latest data
    """
    fig_hist = go.Figure()

    if page_log_training is not None and len(page_log_training["loss"]) > 0:
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
    prevent_initial_call=True,
)
def update_expval(
    n: int,  # modified_timestamp
    page_log_training: Optional[Dict[str, Any]],  # page_log_storage
) -> go.Figure:  # return type
    """
    Updates the expectation value plot with the latest data from the training log.

    :param n: modified_timestamp from the dcc.Store
    :param page_log_training: data from the dcc.Store
    :return: a go.Figure with the latest data
    """
    fig_expval = go.Figure()

    if page_log_training is not None and len(page_log_training["loss"]) > 0:
        fig_expval.add_scatter(
            x=page_log_training["x"], y=page_log_training["y_hat"], name="Prediction"
        )
        fig_expval.add_scatter(
            x=page_log_training["x"], y=page_log_training["y"], name="Target"
        )

    fig_expval.update_layout(
        title="Output",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="Expectation Value",
        yaxis_range=[-1, 1],
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
        fig_ent_cap.add_scatter(y=page_log_training["ent_cap"])

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
    if page_log_training is None or page_data is None:
        raise PreventUpdate()

    if len(page_log_training["loss"]) > page_data["steps"]:
        page_log_training = reset_log()

    instructor = Instructor(
        main_data["number_qubits"],
        main_data["number_layers"],
        n_freqs=page_data["n_freqs"],
        seed=main_data["seed"],
        circuit_type=main_data["circuit_type"],
        data_reupload=main_data["data_reupload"],
    )

    page_log_training["params"], cost, pred = instructor.step(
        page_log_training["params"], page_data["noise_params"]
    )

    page_log_training["loss"].append(cost.item())
    page_log_training["y_hat"] = pred
    page_log_training["x"] = instructor.x_d
    page_log_training["y"] = instructor.y_d

    data = instructor.calc_hist(
        params=page_log_training["params"],
        noise_params=page_data["noise_params"],
    )

    page_log_training["X"] = np.arange(
        -len(data) // 2 + 1, len(data) // 2 + 1, 1
    ).tolist()
    page_log_training["steps"] = [i for i in range(len(page_log_training["loss"]))]
    page_log_training["Y"].append(data.tolist())

    if main_data["number_qubits"] > 1:
        instructor.model.params = page_log_training["params"]
        ent_cap = instructor.meyer_wallach(
            noise_params=page_data["noise_params"],
        )

        page_log_training["ent_cap"].append(ent_cap)

    return page_log_training
