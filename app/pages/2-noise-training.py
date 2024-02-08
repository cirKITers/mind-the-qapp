import dash

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
                dcc.Store(id="storage-noise-training-viz", storage_type="session"),
                dcc.Store(id="storage-noise-training-proc", storage_type="session"),
                dbc.Label("Bit-Flip Probability"),
                dcc.Slider(0, 1, 0.1, value=0, id="bit-flip-prob"),
                dbc.Label("Phase Flip Probability"),
                dcc.Slider(0, 1, 0.1, value=0, id="phase-flip-prob"),
                dbc.Label("Amplitude Damping Probability"),
                dcc.Slider(0, 1, 0.1, value=0, id="amplitude-damping-prob"),
                dbc.Label("Phase Damping Probability"),
                dcc.Slider(0, 1, 0.1, value=0, id="phase-damping-prob"),
                dbc.Label("Depolarization Probability"),
                dcc.Slider(0, 1, 0.1, value=0, id="depolarization-prob"),
                html.Div(
                    [
                        dbc.Button(
                            "Start Training",
                            id="training-button",
                        ),
                        # dcc.Loading(
                        #     id="loading-2",
                        #     children=[html.Div([html.Div(id="loading-output-2")])],
                        #     type="circle",
                        # ),
                        dcc.Interval(
                            id="interval-component",
                            interval=1 * 1000,  # in milliseconds
                            n_intervals=0,
                        ),
                    ],
                ),
            ],
            id="input-container",
        ),
        html.Div(
            [
                dcc.Graph(id="fig-training-hist", style={"display": "inline-block"}),
                dcc.Graph(id="fig-training-expval", style={"display": "inline-block"}),
                dcc.Graph(id="fig-training-metric"),
            ],
            id="output-container",
        ),
    ]
)


instructor = Instructor(2, 4)


@callback(
    Output("storage-noise-training-viz", "data"),
    [
        Input("bit-flip-prob", "value"),
        Input("phase-flip-prob", "value"),
        Input("amplitude-damping-prob", "value"),
        Input("phase-damping-prob", "value"),
        Input("depolarization-prob", "value"),
    ],
    State("storage-noise-training-viz", "data"),
)
def on_preference_changed(bf, pf, ad, pd, dp, data):

    # Give a default data dict with 0 clicks if there's no data.
    data = dict(bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

    return data


@callback(
    Output("fig-training-hist", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_hist(n):
    # if len(instructor.y) == 0:
    #     return None
    return None
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
    Output("fig-training-expval", "figure"),
    Input("interval-component", "n_intervals"),
    State("bit-flip-prob", "value"),
    State("phase-flip-prob", "value"),
    State("amplitude-damping-prob", "value"),
    State("phase-damping-prob", "value"),
    State("depolarization-prob", "value"),
)
def update_expval(n, bf, pf, ad, pd, dp):
    fig_expval = go.Figure()
    # instructor.y = rng.random(size=len(instructor.y_d))
    return None

    if len(instructor.pred) > 0:
        fig_expval.add_scatter(x=instructor.x_d, y=instructor.pred)
    else:
        y = instructor.forward(
            instructor.x_d,
            bf=bf,
            pf=pf,
            ad=ad,
            pd=pd,
            dp=dp,
        )

        if hasattr(y, "_value"):
            y = y._value
        fig_expval.add_scatter(
            x=instructor.x_d,
            y=y,
        )

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
    Output("fig-training-metric", "figure"),
    Input("interval-component", "n_intervals"),
    State("storage-noise-training-proc", "data"),
)
def update_loss(n, page_log):
    page_log = page_log or {"loss": [], "weights": []}

    fig_expval = go.Figure()
    if len(page_log["loss"]) > 0:
        fig_expval.add_scatter(y=page_log["loss"])

    fig_expval.update_layout(
        title="Loss",
        template="simple_white",
        xaxis_title="Step",
        yaxis_title="Loss",
        xaxis_range=[0, 30],
        autosize=False,
        # width=2400,
        height=400,
    )

    return fig_expval


@callback(
    Output("storage-noise-training-proc", "data", allow_duplicate=True),
    Input("storage-noise-training-proc", "modified_timestamp"),
    State("storage-noise-training-proc", "data"),
    prevent_initial_call=True,
)
def pong(_, data):
    if len(data["loss"]) == 30:
        raise PreventUpdate()
    print(len(data["loss"]))
    return data


@callback(
    Output("storage-noise-training-proc", "data"),
    [
        Input("training-button", "n_clicks"),
        Input("storage-noise-training-proc", "data"),
    ],
    [
        State("storage-noise-training-viz", "data"),
        State("storage-main", "data"),
    ],
    prevent_initial_call=True,
)
def training(n, page_log, page_data, main_data):
    page_log = page_log or {"loss": [], "weights": []}

    if len(page_log["loss"]) == 30:
        page_log["loss"] = []
        page_log["weights"] = []

    bf, pf, ad, pd, dp = (
        page_data["bf"],
        page_data["pf"],
        page_data["ad"],
        page_data["pd"],
        page_data["dp"],
    )

    instructor = Instructor(main_data["niq"], main_data["nil"], seed=main_data["seed"])

    page_log["weights"], cost = instructor.step(
        page_log["weights"], bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
    )
    page_log["loss"].append(cost.item())

    return page_log
