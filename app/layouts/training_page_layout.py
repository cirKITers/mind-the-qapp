from dash import (
    dcc,
    html,
)
import dash_bootstrap_components as dbc

DEFAULT_N_STEPS = 50
DEFAULT_N_FREQS = 4
DEFAULT_STEPSIZE = 0.01
MAX_N_FREQS = 11


def generate_coefficient_sliders(n_freqs=DEFAULT_N_FREQS):
    """Generate sliders for Fourier series coefficients."""
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Input(
                        type="number",
                        id=f"coef-input-{i}",
                        value=0.5,
                        min=0,
                        max=1,
                        size="sm",
                    ),
                    html.Div(
                        dcc.Slider(
                            min=0,
                            max=1,
                            value=0.5,
                            vertical=True,
                            marks={
                                **{i / 10: str(i / 10) for i in range(1, 10)},
                                0: "0.0",
                                1: "1.0",
                            },
                            id=f"coef-slider-{i}",
                            updatemode="mouseup",  # 'drag' for continuous updates
                        ),
                    ),
                ],
                id=f"coef-col-{i}",
                style={
                    "minWidth": "50px",
                    "maxWidth": "70px",
                    "width": "auto",
                    "visibility": "visible" if i < n_freqs else "hidden",
                    "display": "block" if i < n_freqs else "none",
                },
            )
            for i in range(MAX_N_FREQS)
        ],
        className="g-1",
    )


# Create popover content
coefficient_popover = dbc.Popover(
    [
        dbc.PopoverHeader("Fourier Series Coefficients"),
        dbc.PopoverBody(
            html.Div(
                id="coefficient-sliders-container",
                children=generate_coefficient_sliders(),
                style={"display": "flex", "justifyContent": "center", "width": "100%"},
            )
        ),
    ],
    id="coefficient-settings-popover",
    target="open-coefficient-modal",  # ID of the gear button
    trigger="click",  # Show on click
    placement="bottom",  # Show below the button
)

layout = html.Div(
    [
        dcc.Store(id="training-page-storage", storage_type="session"),
        dcc.Store(id="training-log-storage", storage_type="session"),
        coefficient_popover,
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
                                                        "display": "block",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Input(
                                                            type="number",
                                                            min=1,
                                                            max=MAX_N_FREQS,
                                                            step=1,
                                                            value=DEFAULT_N_FREQS,
                                                            id="training-freqs-numeric-input",
                                                            style={
                                                                "marginRight": "10px"
                                                            },
                                                        ),
                                                        dbc.Button(
                                                            html.Img(
                                                                src="/assets/gear-solid.svg",
                                                                style={
                                                                    "width": "24px",
                                                                    "height": "24px",
                                                                    "filter": "invert(1)",
                                                                },
                                                            ),
                                                            id="open-coefficient-modal",
                                                            style={
                                                                "aspect-ratio": "1",
                                                                "padding": "6px",
                                                                "display": "flex",
                                                                "align-items": "center",
                                                                "justify-content": "center",
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "alignItems": "center",
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
                                                        "display": "block",  # Changed to block for stacking
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
                                                        "width": "5vw",
                                                        "display": "block",  # Changed to block for stacking
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
                                                        "display": "block",
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
                                                        "display": "block",
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
