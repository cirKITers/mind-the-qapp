from dash import (
    dcc,
    html,
)
import dash_bootstrap_components as dbc

layout = html.Div(
    [
        dcc.Store(id="expr-page-storage", storage_type="session"),
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
                                                html_for="expr-param-sample-pairs-input",
                                            ),
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                type="number",
                                                min=100,
                                                max=1000,
                                                step=1,
                                                value=200,
                                                id="expr-param-sample-pairs-input",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Input Samples",
                                                html_for="expr-samples-input",
                                            ),
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                type="number",
                                                min=2,
                                                max=100,
                                                step=1,
                                                value=5,
                                                id="expr-samples-input",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Histogram Bins",
                                                html_for="expr-histogram-bins-input",
                                            ),
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                type="number",
                                                min=10,
                                                max=500,
                                                step=1,
                                                value=20,
                                                id="expr-histogram-bins-input",
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Meyer-Wallach Entangling Capability",
                                                html_for="expr-ent-cap-badge",
                                            ),
                                        ),
                                        dbc.Col(
                                            html.H4(
                                                dbc.Badge(
                                                    "0.0",
                                                    color="primary",
                                                    pill=True,
                                                    className="me-1",
                                                    id="expr-ent-cap-badge",
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
                                                id="expr-phase-damping-prob-slider",
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
                                                id="expr-depolarization-prob-slider",
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
                                                id="expr-bit-flip-prob-slider",
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
                                                id="expr-phase-flip-prob-slider",
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
                                                id="expr-amplitude-damping-prob-slider",
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
                            id="expr-hist-figure",
                            style={
                                "display": "inline-block",
                                "height": "50vh",
                                "width": "100%",
                            },
                        ),
                        dcc.Graph(
                            id="expr-kl-figure",
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
                            id="expr-haar-figure",
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
