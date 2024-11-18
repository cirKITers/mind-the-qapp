import dash_bootstrap_components as dbc
from dash import html, dcc, html, page_container
from qml_essentials.ansaetze import Ansaetze

DEFAULT_N_QUBITS = 1
DEFAULT_N_LAYERS = 1
DEFAULT_SEED = 100
DEFAULT_DATA_REUPLOAD = True
DEFAULT_ANSATZ = "No_Ansatz"

sidebar_top = [
    dcc.Store(id="main-storage", storage_type="session"),
    html.Div(
        [
            html.H1(
                "Mind",
            ),
            html.H2(
                "the",
                style={
                    "padding-left": "6px",
                },
            ),
            html.Span(
                [
                    html.Img(
                        src="assets/underground-sign.svg",
                        width="42",
                        style={
                            "display": "inline-block",
                            "padding-right": "5px",
                            "padding-bottom": "20px",
                        },
                        className="rotate45",
                    ),
                    html.H1("App", style={"display": "inline-block"}),
                ]
            ),
        ],
        className="infoBox",
    ),
    html.Hr(),
]
sidebar_bottom = [
    html.Div(
        [
            html.Div(
                [
                    dbc.Label("# of Qubits (0-10)"),
                    dbc.Input(
                        type="number",
                        min=0,
                        max=10,
                        step=1,
                        value=DEFAULT_N_QUBITS,
                        id="main-qubits-input",
                    ),
                ],
                className="numeric-input",
            ),
            html.Div(
                [
                    dbc.Label("# of Layers (0-10)"),
                    dbc.Input(
                        type="number",
                        min=0,
                        max=10,
                        step=1,
                        value=DEFAULT_N_LAYERS,
                        id="main-layers-input",
                    ),
                ],
                className="numeric-input",
            ),
            html.Div(
                [
                    dbc.Label("Circuit Type"),
                    dbc.Select(
                        options=[
                            {
                                "label": fct.__name__.replace("_", " ").title(),
                                "value": fct.__name__,
                            }
                            for fct in Ansaetze.get_available()
                        ],
                        placeholder=DEFAULT_ANSATZ,
                        required=True,
                        id="main-circuit-ident-select",
                    ),
                ],
                className="numeric-input",
            ),
            html.Div(
                [
                    dbc.Label("Data-Reupload"),
                    dbc.Switch(
                        id="main-dru-switch",
                        value=DEFAULT_DATA_REUPLOAD,
                        className="fs-4",
                    ),
                ],
            ),
            # html.Div(
            #     [
            #         dbc.Label("Trainable Freqs."),
            #         dbc.Switch(id="switch-tffm", value=False),
            #     ],
            # ),
            html.Div(
                [
                    dbc.Label("Seed"),
                    dbc.Input(
                        type="number",
                        min=100,
                        max=999,
                        step=1,
                        value=DEFAULT_SEED,
                        id="main-seed-input",
                    ),
                ],
                className="numeric-input",
            ),
        ],
        className="preferencesBox",
    ),
    html.Hr(),
    html.Div(
        [
            dbc.Spinner(
                [html.H6("", id="main-loading-state")],
                color="primary",
                type="grow",
                id="main-loading-spinner",
            )
        ],
        className="spinnerBox",
    ),
]

content = html.Div(
    [
        page_container,
    ],
    className="content",
    id="page-content",
)
