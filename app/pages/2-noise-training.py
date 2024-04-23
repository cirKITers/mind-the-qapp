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
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate


from utils.instructor import Instructor
from utils.entangling import EntanglingCapability_Sampler

import dash_bootstrap_components as dbc

dash.register_page(__name__, name="Noise Training")

layout = html.Div(
    [
        dcc.Store(id="storage-noise-training-viz", storage_type="session"),
        dcc.Store(id="storage-noise-training-proc", storage_type="session"),
        dcc.Store(id="storage-noise-hist-proc", storage_type="session"),
        dcc.Interval(
            id="interval-component",
            interval=1 * 1000,  # in milliseconds
            n_intervals=0,
        ),
        html.Div(
            [
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Label("Bit-Flip Probability"),
                                dcc.Slider(
                                    0, 0.5, 0.05, value=0, id="bit-flip-prob-training"
                                ),
                                dbc.Label("Phase Flip Probability"),
                                dcc.Slider(
                                    0, 0.5, 0.05, value=0, id="phase-flip-prob-training"
                                ),
                                dbc.Label(
                                    "Amplitude Damping Probability",
                                ),
                                dcc.Slider(
                                    0,
                                    0.5,
                                    0.05,
                                    value=0,
                                    id="amplitude-damping-prob-training",
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
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Start Training",
                                                    id="training-button",
                                                ),
                                            ],
                                            style={
                                                # "height": "1vh",
                                                # "width": "10vh",
                                                "display": "inline-block",
                                            },
                                        ),
                                        html.Div(
                                            [dbc.Label("Steps:")],
                                            style={
                                                "display": "inline-block",
                                                "padding": "0 20px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                dbc.Input(
                                                    type="number",
                                                    min=1,
                                                    max=101,
                                                    step=1,
                                                    value=10,
                                                    id="numeric-input-steps",
                                                ),
                                            ],
                                            style={
                                                "width": "10vw",
                                                "display": "inline-block",
                                            },
                                        ),
                                    ],
                                    style={"padding": "15px"},
                                ),
                                dbc.Label("Phase Damping Probability"),
                                dcc.Slider(
                                    0,
                                    0.5,
                                    0.05,
                                    value=0,
                                    id="phase-damping-prob-training",
                                ),
                                dbc.Label("Depolarization Probability"),
                                dcc.Slider(
                                    0,
                                    0.5,
                                    0.05,
                                    value=0,
                                    id="depolarization-prob-training",
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
                            id="fig-training-hist",
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
                html.Div(
                    [
                        dcc.Graph(
                            id="fig-training-expval",
                            style={
                                "display": "inline-block",
                                "height": "40vh",
                                "width": "100%",
                            },
                        ),
                        dcc.Graph(
                            id="fig-training-metric",
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
    Output("storage-noise-training-viz", "data", allow_duplicate=True),
    Input("storage-main", "modified_timestamp"),
    State("storage-noise-training-viz", "data"),
    prevent_initial_call=True,
)
def update_page_data(_, page_data):
    return page_data


@callback(
    [
        Output("storage-noise-training-viz", "data"),
        Output("storage-noise-hist-proc", "data"),
    ],
    [
        Input("bit-flip-prob-training", "value"),
        Input("phase-flip-prob-training", "value"),
        Input("amplitude-damping-prob-training", "value"),
        Input("phase-damping-prob-training", "value"),
        Input("depolarization-prob-training", "value"),
        Input("numeric-input-steps", "value"),
    ],
)
def on_preference_changed(bf, pf, ad, pd, dp, steps):

    # Give a default data dict with 0 clicks if there's no data.
    page_data = dict(bf=bf, pf=pf, ad=ad, pd=pd, dp=dp, steps=steps)
    page_log_hist = {"x": [], "y": [], "z": []}

    return page_data, page_log_hist


@callback(
    [
        Output("fig-training-hist", "figure"),
        Output("storage-noise-hist-proc", "data", allow_duplicate=True),
    ],
    Input("storage-noise-training-proc", "modified_timestamp"),
    [
        State("storage-noise-training-proc", "data"),
        State("storage-noise-hist-proc", "data"),
        State("storage-noise-training-viz", "data"),
        State("storage-main", "data"),
    ],
    prevent_initial_call=True,
)
def update_hist(n, page_log_training, page_log_hist, page_data, main_data):
    fig_hist = go.Figure()

    if page_log_hist is None or len(page_log_training["loss"]) == 0:
        page_log_hist = {"x": [], "y": [], "z": []}
    else:
        instructor = Instructor(
            main_data["number_qubits"],
            main_data["number_layers"],
            seed=main_data["seed"],
            circuit_type=main_data["circuit_type"],
            data_reupload=main_data["data_reupload"],
        )

        bf, pf, ad, pd, dp = (
            page_data["bf"],
            page_data["pf"],
            page_data["ad"],
            page_data["pd"],
            page_data["dp"],
        )

        data_len, data = instructor.calc_hist(
            page_log_training["weights"],
            bf=bf,
            pf=pf,
            ad=ad,
            pd=pd,
            dp=dp,
        )

        page_log_hist["x"] = np.arange(-data_len // 2 + 1, data_len // 2 + 1, 1)
        page_log_hist["y"] = [i for i in range(len(page_log_training["loss"]))]
        page_log_hist["z"].append(data["comb"][0].tolist())

        fig_hist.add_surface(
            x=np.array(page_log_hist["x"]),
            y=np.array(page_log_hist["y"]),
            z=np.array(page_log_hist["z"]),
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
            up=dict(x=0, y=0, z=1),
            center=dict(x=0.1, y=0, z=0),
            eye=dict(x=1.15, y=2.05, z=1.05),
        ),
        coloraxis_showscale=False,
    )

    return fig_hist, page_log_hist


@callback(
    Output("fig-training-expval", "figure"),
    Input("storage-noise-training-proc", "modified_timestamp"),
    [
        State("storage-noise-training-proc", "data"),
        State("storage-noise-training-viz", "data"),
        State("storage-main", "data"),
    ],
    prevent_initial_call=True,
)
def update_expval(n, page_log_training, page_data, main_data):
    fig_expval = go.Figure()

    if len(page_log_training["loss"]) > 0:
        instructor = Instructor(
            main_data["number_qubits"],
            main_data["number_layers"],
            seed=main_data["seed"],
            circuit_type=main_data["circuit_type"],
            data_reupload=main_data["data_reupload"],
        )

        bf, pf, ad, pd, dp = (
            page_data["bf"],
            page_data["pf"],
            page_data["ad"],
            page_data["pd"],
            page_data["dp"],
        )

        y_pred = instructor.forward(
            instructor.x_d,
            weights=page_log_training["weights"],
            bf=bf,
            pf=pf,
            ad=ad,
            pd=pd,
            dp=dp,
        )

        fig_expval.add_scatter(x=instructor.x_d, y=y_pred, name="Prediction")
        fig_expval.add_scatter(x=instructor.x_d, y=instructor.y_d, name="Target")
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
    Input("storage-noise-training-proc", "modified_timestamp"),
    [
        State("storage-noise-training-proc", "data"),
        State("storage-noise-training-viz", "data"),
    ],
    prevent_initial_call=True,
)
def update_ent_cap(n, page_log_training, data):
    fig_ent_cap = go.Figure()
    if len(page_log_training["loss"]) > 0:
        fig_ent_cap.add_scatter(y=page_log_training["ent_cap"])

    fig_ent_cap.update_layout(
        title="Entangling Capability",
        template="simple_white",
        xaxis_title="Step",
        yaxis_title="Entangling Capability",
        xaxis_range=[0, data["steps"]],
        autosize=False,
    )

    return fig_ent_cap

@callback(
    Output("fig-training-metric", "figure"),
    Input("storage-noise-training-proc", "modified_timestamp"),
    [
        State("storage-noise-training-proc", "data"),
        State("storage-noise-training-viz", "data"),
    ],
    prevent_initial_call=True,
)
def update_loss(n, page_log_training, data):
    fig_expval = go.Figure()
    if len(page_log_training["loss"]) > 0:
        fig_expval.add_scatter(y=page_log_training["loss"])

    fig_expval.update_layout(
        title="Loss",
        template="simple_white",
        xaxis_title="Step",
        yaxis_title="Loss",
        xaxis_range=[0, data["steps"]],
        autosize=False,
    )

    return fig_expval


@callback(
    [
        Output("storage-noise-training-proc", "data", allow_duplicate=True),
        Output("training-button", "disabled", allow_duplicate=True),
    ],
    Input("training-button", "n_clicks"),
    prevent_initial_call=True,
)
def trigger_training(_):
    page_log = {"loss": [], "weights": [], "ent_cap": []}

    return [page_log, True]


@callback(
    Output("training-button", "disabled", allow_duplicate=True),
    Input("storage-noise-training-proc", "modified_timestamp"),
    [
        State("storage-noise-training-proc", "data"),
        State("storage-noise-training-viz", "data"),
    ],
    prevent_initial_call=True,
)
def stop_training(_, page_log_training, page_data):
    if len(page_log_training["loss"]) <= page_data["steps"]:
        raise PreventUpdate()

    return False


@callback(
    Output("storage-noise-training-proc", "data", allow_duplicate=True),
    Input("storage-noise-training-proc", "modified_timestamp"),
    [
        State("storage-noise-training-proc", "data"),
        State("storage-noise-training-viz", "data"),
    ],
    prevent_initial_call=True,
)
def pong(_, page_log_training, data):
    if page_log_training is None:
        raise PreventUpdate()

    if len(page_log_training["loss"]) > data["steps"]:
        raise PreventUpdate()
    return page_log_training


@callback(
    Output("storage-noise-training-proc", "data"),
    [
        Input("storage-noise-training-proc", "data"),
    ],
    [
        State("storage-noise-training-viz", "data"),
        State("storage-main", "data"),
    ],
    prevent_initial_call=True,
)
def training(page_log_training, page_data, main_data):
    if page_log_training is None:
        raise PreventUpdate()

    if len(page_log_training["loss"]) > page_data["steps"]:
        page_log_training["loss"] = []
        page_log_training["weights"] = []
        page_log_training["ent_cap"] = []

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

    page_log_training["weights"], cost = instructor.step(
        page_log_training["weights"], bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
    )
    page_log_training["loss"].append(cost.item())

    ent_sampler = EntanglingCapability_Sampler(
        main_data["number_qubits"],
        main_data["number_layers"],
        main_data["seed"],
        main_data["circuit_type"],
        main_data["data_reupload"],
    )

    ent_cap = ent_sampler.calculate_entangling_capability(
        10, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp, params=page_log_training["weights"]
    )

    page_log_training["ent_cap"].append(ent_cap)

    return page_log_training
