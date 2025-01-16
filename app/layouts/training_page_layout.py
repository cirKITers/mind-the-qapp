from dash import (
    dcc,
    html,
)
import dash_bootstrap_components as dbc

DEFAULT_N_STEPS = 50
DEFAULT_N_FREQS = 4
DEFAULT_STEPSIZE = 0.01

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
                                                        "width": "4vw",
                                                        "display": "inline-block",
                                                        "padding-left": "8px",
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
                                                            max=801,
                                                            step=1,
                                                            value=DEFAULT_N_STEPS,
                                                            id="training-steps-numeric-input",
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "4vw",
                                                        "display": "inline-block",
                                                        "padding-left": "8px",
                                                    },
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label("Stepsize: "),
                                                    ],
                                                    style={
                                                        "display": "inline-block",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Input(
                                                            type="number",
                                                            min=0.001,
                                                            max=0.1,
                                                            step=0.001,
                                                            value=DEFAULT_STEPSIZE,
                                                            id="training-stepsize-numeric-input",
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "5vw",
                                                        "display": "inline-block",
                                                        "padding-left": "8px",
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
                                    style={"padding-top": "10px"},
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
