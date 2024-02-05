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


from utils.instructor import Instructor

import dash_bootstrap_components as dbc

dash.register_page(__name__, name="Noise Training")

layout = html.Div(
    [
        html.Div(
            id="input-container",
            children=[
                html.Label("Bit-Flip Probability", htmlFor="bit-flip-prob"),
                dcc.Slider(0, 1, 0.1, value=0, id="bit-flip-prob"),
                html.Label("Phase Flip Probability", htmlFor="phase-flip-prob"),
                dcc.Slider(0, 1, 0.1, value=0, id="phase-flip-prob"),
                html.Label(
                    "Amplitude Damping Probability", htmlFor="amplitude-damping-prob"
                ),
                dcc.Slider(0, 1, 0.1, value=0, id="amplitude-damping-prob"),
                html.Label("Phase Damping Probability", htmlFor="phase-damping-prob"),
                dcc.Slider(0, 1, 0.1, value=0, id="phase-damping-prob"),
                html.Label("Depolarization Probability", htmlFor="depolarization-prob"),
                dcc.Slider(0, 1, 0.1, value=0, id="depolarization-prob"),
                html.Div(
                    [
                        dbc.Button(
                            "Start Training",
                            id="training-button",
                        ),
                        dcc.Loading(
                            id="loading-2",
                            children=[html.Div([html.Div(id="loading-output-2")])],
                            type="circle",
                        ),
                        dcc.Interval(
                            id="interval-component",
                            interval=1 * 1000,  # in milliseconds
                            n_intervals=0,
                        ),
                    ],
                ),
            ],
            style={
                "margin-left": "100px",
                "margin-right": "100px",
                "margin-top": "35px",
                "margin-bottom": "35px",
            },
        ),
        html.Div(
            id="output-container",
            children=[
                dcc.Graph(id="fig-training-hist", style={"display": "inline-block"}),
                dcc.Graph(id="fig-training-expval", style={"display": "inline-block"}),
                dcc.Graph(id="fig-training-metric"),
            ],
            style={
                "margin-left": "100px",
                "margin-right": "100px",
            },
        ),
    ]
)


instructor = Instructor(2, 4)


@callback(
    Output("fig-training-hist", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_hist(n):
    # if len(instructor.y) == 0:
    #     return None

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
)
def update_expval(n):
    fig_expval = go.Figure()

    if len(instructor.loss) > 0:
        fig_expval.add_scatter(y=instructor.loss)

    fig_expval.update_layout(
        title="Loss",
        template="simple_white",
        xaxis_title="Step",
        yaxis_title="Loss",
        xaxis_range=[0, instructor.steps],
        autosize=False,
        width=2400,
        height=400,
    )

    return fig_expval


@callback(
    Output("training-button", "disabled"),
    Input("training-button", "n_clicks"),
    State("bit-flip-prob", "value"),
    State("phase-flip-prob", "value"),
    State("amplitude-damping-prob", "value"),
    State("phase-damping-prob", "value"),
    State("depolarization-prob", "value"),
)
def training(n, bf, pf, ad, pd, dp):
    if n is None:
        return None

    instructor.clear_hist()
    _weights = instructor.weights.copy()
    print("Training started")
    for s in range(instructor.steps):
        pred, _weights, cost = instructor.step(
            _weights, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
        )
        instructor.pred = pred
        instructor.loss.append(cost)

        data_len, data = instructor.calc_hist(
            _weights, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
        )

        instructor.append_hist(data["comb"][0])

    print("Training finished")
    return False
