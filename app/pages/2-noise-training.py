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

import dash_bootstrap_components as dbc

dash.register_page(__name__, name="Noise Training")

layout = html.Div(
    [
        html.Div(
            [
                dcc.Store(id="storage-noise-viz", storage_type="session"),
                dcc.Store(id="storage-noise-training-viz", storage_type="session"),
                dcc.Store(id="storage-noise-training-proc", storage_type="session"),
                dcc.Store(id="storage-noise-hist-proc", storage_type="session"),
                html.Div(
                    [
                        dbc.Label("Bit-Flip Probability"),
                        dcc.Slider(0, 0.5, 0.05, value=0, id="bit-flip-prob-training"),
                        dbc.Label("Phase Flip Probability"),
                        dcc.Slider(
                            0, 0.5, 0.05, value=0, id="phase-flip-prob-training"
                        ),
                        dbc.Label(
                            "Amplitude Damping Probability",
                        ),
                        dcc.Slider(
                            0, 0.5, 0.05, value=0, id="amplitude-damping-prob-training"
                        ),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dbc.Label("Phase Damping Probability"),
                        dcc.Slider(
                            0, 0.5, 0.05, value=0, id="phase-damping-prob-training"
                        ),
                        dbc.Label("Depolarization Probability"),
                        dcc.Slider(
                            0, 0.5, 0.05, value=0, id="depolarization-prob-training"
                        ),
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
                                        "padding": "0 10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        dbc.Input(
                                            type="number",
                                            min=1,
                                            max=50,
                                            step=1,
                                            value=10,
                                            id="numeric-input-steps",
                                        ),
                                    ],
                                    style={"width": "10vh", "display": "inline-block"},
                                ),
                                # dcc.Loading(
                                #     id="loading-2",
                                #     children=[html.Div([html.Div(id="loading-output-2")])],
                                #     type="circle",
                                # ),
                            ],
                        ),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=1 * 1000,  # in milliseconds
                    n_intervals=0,
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
                                "height": "80vh",
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
    Output("storage-noise-training-viz", "data"),
    Output("storage-noise-hist-proc", "data"),
    [
        Input("bit-flip-prob-training", "value"),
        Input("phase-flip-prob-training", "value"),
        Input("amplitude-damping-prob-training", "value"),
        Input("phase-damping-prob-training", "value"),
        Input("depolarization-prob-training", "value"),
        Input("numeric-input-steps", "value"),
        Input("storage-main", "modified_timestamp"),
    ],
    State("storage-noise-training-viz", "data"),
    State("storage-main", "data"),
)
def on_preference_changed(bf, pf, ad, pd, dp, steps, _, page_data, main_data):

    # Give a default data dict with 0 clicks if there's no data.
    page_data = dict(bf=bf, pf=pf, ad=ad, pd=pd, dp=dp, steps=steps)
    page_log_hist = {"x": [], "y": [], "z": []}

    return page_data, page_log_hist


@callback(
    Output("fig-training-hist", "figure"),
    Output("storage-noise-hist-proc", "data", allow_duplicate=True),
    Input("storage-noise-training-proc", "modified_timestamp"),
    State("storage-noise-training-proc", "data"),
    State("storage-noise-hist-proc", "data"),
    State("storage-noise-training-viz", "data"),
    State("storage-main", "data"),
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
        )
    fig_hist.update_layout(
        title="Histogram (Absolute Value)",
        template="simple_white",
        # margin=dict(l=65, r=50, b=65, t=90),
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
    )

    return fig_hist, page_log_hist


@callback(
    Output("fig-training-expval", "figure"),
    Input("storage-noise-training-proc", "modified_timestamp"),
    State("storage-noise-training-proc", "data"),
    State("storage-noise-training-viz", "data"),
    State("storage-main", "data"),
    prevent_initial_call=True,
)
def update_expval(n, page_log_training, page_data, main_data):
    fig_expval = go.Figure()

    if len(page_log_training["loss"]) > 0:
        instructor = Instructor(
            main_data["number_qubits"],
            main_data["number_layers"],
            seed=main_data["seed"],
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

        fig_expval.add_scatter(
            x=instructor.x_d,
            y=y_pred,
        )
        fig_expval.add_scatter(
            x=instructor.x_d,
            y=instructor.y_d,
        )
    fig_expval.update_layout(
        title="Output",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="Expectation Value",
        yaxis_range=[-1, 1],
    )

    return fig_expval


@callback(
    Output("fig-training-metric", "figure"),
    Input("storage-noise-training-proc", "modified_timestamp"),
    State("storage-noise-training-proc", "data"),
    State("storage-noise-training-viz", "data"),
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
    page_log = {"loss": [], "weights": []}

    return [page_log, True]


@callback(
    Output("training-button", "disabled", allow_duplicate=True),
    Input("storage-noise-training-proc", "modified_timestamp"),
    State("storage-noise-training-proc", "data"),
    State("storage-noise-training-viz", "data"),
    prevent_initial_call=True,
)
def stop_training(_, page_log_training, page_data):
    if len(page_log_training["loss"]) <= page_data["steps"]:
        raise PreventUpdate()

    return False


@callback(
    Output("storage-noise-training-proc", "data", allow_duplicate=True),
    Input("storage-noise-training-proc", "modified_timestamp"),
    State("storage-noise-training-proc", "data"),
    State("storage-noise-training-viz", "data"),
    prevent_initial_call=True,
)
def pong(_, page_log_training, data):
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

    if len(page_log_training["loss"]) > page_data["steps"]:
        page_log_training["loss"] = []
        page_log_training["weights"] = []

    bf, pf, ad, pd, dp = (
        page_data["bf"],
        page_data["pf"],
        page_data["ad"],
        page_data["pd"],
        page_data["dp"],
    )

    instructor = Instructor(
        main_data["number_qubits"], main_data["number_layers"], seed=main_data["seed"]
    )

    page_log_training["weights"], cost = instructor.step(
        page_log_training["weights"], bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
    )
    page_log_training["loss"].append(cost.item())

    return page_log_training
