import dash
from dash import (
    dcc,
    html,
)

dash.register_page(__name__, name="Home", path="/")

layout = html.Div(
    [
        dcc.Markdown(
            """
        # Mind the QApp
        ## About

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

        ## Getting started

        On the left you can see two pages:

        - [Training](/1-training)
        - [Expressibility](/2-expressibility)

        The first one allows you to run a training of a data-reuploading enabled model on a simple
        Fourier series dataset. During training you can observe how the loss changes and the model
        output gets closer to the target. Play around with the number of qubits and layers as well as
        the circuit type to see which model performs best. Besides the loss and the model output, you
        can also observe how the Fourier spectrum changes during training (3D Histogram) and how the
        entangling capability of the chosen Ansatz changes.

        On the second page you can evaluate the expressibility of a chosen Ansatz and how it compares
        to the Haar measure.

        Both pages offer settings to adjust various types of noise that apply directly to the model
        used in the experiments.
        """
        ),
    ]
)
