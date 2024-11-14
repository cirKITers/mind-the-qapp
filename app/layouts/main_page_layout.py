from dash import (
    dcc,
    html,
)
import dash_bootstrap_components as dbc

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Stack(
                            [
                                html.Div(
                                    [
                                        html.H2("About", className="display-3"),
                                        html.Hr(className="my-2"),
                                        dcc.Markdown(
                                            """
                                This application serves a visualization and teaching purpose for simple training scenarios to
                                study various effects and problems in QML. Characteristic parameters of a QML model can be
                                adjusted:

                                - number of qubits
                                - number of layers
                                - type of ansatz
                                - data re-uploading [1](https://arxiv.org/abs/2008.08605)

                                Furthermore, as noise is a non-negligible factor in Quantum Computing, the application allows
                                the user to adjust the strength of various types of noise. This enables to study the impact of
                                noise on the Fourier spectrum, which can be represented with the chosen ansatz
                                [1](https://arxiv.org/abs/2008.08605), both with a fixed set of parameters and within a training
                                scenario.

                                We can also track the entangling capability [2](https://arxiv.org/abs/quant-ph/0305094) of the
                                chosen Ansatz over the training period, so that one can evaluate the effect of parameterized
                                entangling gates on the training performance.

                                The entanglement and expressibility of [3](https://arxiv.org/abs/1905.10876) can also be analyzed
                                in another part of the application, distinct from the training routine, where we present a
                                visualization of expressibility with respect to the input value. This is done for randomly
                                sampled parameter values, where the distance to the Haar measure is indicated by the
                                Kullback-Leibler divergence, as in [3](https://arxiv.org/abs/1905.10876).

                                To improve the performance, we utilize caching strategies.

                                """
                                        ),
                                    ],
                                    className="h-100 p-5 text-white bg-secondary rounded-3",
                                ),
                            ],
                            gap=3,
                        ),
                    ],
                    width={"offset": 0.5},
                ),
                dbc.Col(
                    [
                        dbc.Stack(
                            [
                                html.Div(
                                    [
                                        html.H2(
                                            "Getting started", className="display-3"
                                        ),
                                        html.Hr(className="my-2"),
                                        dcc.Markdown(
                                            """
                                On the left you can see two pages:

                                - Training
                                - Expressibility

                                where you can find details in the boxes below.

                                Also on the left, you can find fields to adjust the number
                                of qubits, the number of layers, and the type of ansatz.
                                Via the switch you can enable data re-uploading.
                                """
                                        ),
                                    ],
                                    className="h-100 p-5 bg-light text-dark border rounded-3",
                                ),
                                html.Div(
                                    [
                                        html.H2("Training", className="display-3"),
                                        html.Hr(className="my-2"),
                                        dcc.Markdown(
                                            """
                                The first one allows you to run a training of a data-reuploading enabled model on a simple
                                Fourier series dataset. During training you can observe how the loss changes and the model
                                output gets closer to the target. Play around with the number of qubits and layers as well as
                                the circuit type to see which model performs best. Besides the loss and the model output, you
                                can also observe how the Fourier spectrum changes during training (3D Histogram) and how the
                                entangling capability of the chosen Ansatz changes.

                                """
                                        ),
                                        dbc.Button(
                                            "Go to Training page",
                                            outline=True,
                                            color="primary",
                                            className="me-1",
                                            href="/1-training",
                                        ),
                                    ],
                                    className="h-100 p-5 bg-light text-dark border rounded-3",
                                ),
                                html.Div(
                                    [
                                        html.H2(
                                            "Expressibility", className="display-3"
                                        ),
                                        html.Hr(className="my-2"),
                                        dcc.Markdown(
                                            """
                                On the second page you can evaluate the expressibility of a chosen Ansatz and how it compares
                                to the Haar measure.

                                Both pages offer settings to adjust various types of noise that apply directly to the model
                                used in the experiments.

                                """
                                        ),
                                        dbc.Button(
                                            "Go to Expressibility page",
                                            outline=True,
                                            color="primary",
                                            className="me-1",
                                            href="/2-expressibility",
                                        ),
                                    ],
                                    className="h-100 p-5 bg-light text-dark border rounded-3",
                                ),
                            ],
                            gap=3,
                        ),
                    ],
                    width={"offset": 0.5},
                ),
            ]
        )
    ]
)
