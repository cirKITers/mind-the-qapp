from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go

from functools import partial

import pennylane as qml
import pennylane.numpy as np
from pennylane.fourier import coefficients
from pennylane.fourier.visualize import _extract_data_and_labels

app = Dash(__name__)


app.layout = html.Div(
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
            ],
        ),
        html.Div(
            id="output-container",
            children=[
                dcc.Graph(id="fig-hist"),
                dcc.Graph(id="fig-expval"),
            ],
        ),
    ]
)


class Model:
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.mixed", wires=n_qubits)

        self.circuit = qml.QNode(self._circuit, self.dev)

    def pqc(self, w: np.ndarray):
        """
        Creates a Circuit19 ansatz.

        Length of flattened vector must be n_qubits*3-1
        because for >1 qubits there are three gates

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
        """

        w_idx = 0
        for q in range(self.n_qubits):
            qml.RX(w[w_idx], wires=q)
            w_idx += 1
            qml.RZ(w[w_idx], wires=q)
            w_idx += 1

            if q > 0:
                qml.CRX(w[w_idx], wires=[q, (q + 1) % self.n_qubits])
                w_idx += 1

    def iec(self, x: np.ndarray):
        """
        Creates an AngleEncoding using RY gates

        Args:
            x (np.ndarray): length of vector must be 1
        """

        for q in range(self.n_qubits):
            qml.RY(x, wires=q)

    def _circuit(
        self, w: np.ndarray, x: np.ndarray, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0
    ):
        """
        Creates a circuit with noise.
        This involves, Amplitude Damping, Phase Damping and Depolarization.
        The Circuit consists of a PQC and IEC in each layer.

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
            x (np.ndarray): input vector of size 1
            ad (float, optional): Amplitude Damping. Defaults to 0.0.
            pd (float, optional): Phase Damping. Defaults to 0.0.
            dp (float, optional): Depolarization. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        for l in range(self.n_layers):
            self.pqc(w[l])
            self.iec(x)

            for q in range(self.n_qubits):
                qml.BitFlip(bf, wires=q)
                qml.PhaseFlip(pf, wires=q)
                qml.AmplitudeDamping(ad, wires=q)
                qml.PhaseDamping(pd, wires=q)
                qml.DepolarizingChannel(dp, wires=q)

        return qml.expval(qml.PauliZ(0))


rng = np.random.default_rng(100)

n_qubits = 2
n_layers = 4

weights = np.pi * (1 - 2 * rng.random(size=(n_layers, n_qubits * 3 - 1)))

model = Model(n_qubits, n_layers)

x_domain = [-1 * np.pi, 1 * np.pi]  # [-4 * np.pi, 4 * np.pi]
omega_d = np.array([1, 1.2, 3])

n_d = int(np.ceil(2 * np.max(np.abs(x_domain)) * np.max(omega_d)))
print(f"Using {n_d} data points")
x_d = np.linspace(x_domain[0], x_domain[1], n_d)


@callback(
    [Output("fig-hist", "figure"), Output("fig-expval", "figure")],
    [
        Input("bit-flip-prob", "value"),
        Input("phase-flip-prob", "value"),
        Input("amplitude-damping-prob", "value"),
        Input("phase-damping-prob", "value"),
        Input("depolarization-prob", "value"),
    ],
)
def update_output(bf, pf, ad, pd, dp):
    coeffs = coefficients(
        partial(model.circuit, weights, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp),
        1,
        n_qubits * n_layers,
    )
    nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
    data_len = len(data["real"][0])
    data["comb"] = np.sqrt(data["real"] ** 2 + data["imag"] ** 2)

    y_pred = model.circuit(weights, x_d, bf, pf, ad, pd, dp)

    fig_hist = go.Figure()
    fig_expval = go.Figure()

    fig_expval.add_scatter(x=x_d, y=y_pred)
    fig_hist.add_bar(
        x=np.arange(-data_len // 2 + 1, data_len // 2 + 1, 1), y=data["comb"][0]
    )

    fig_expval.update_layout(
        title="Prediction",
        template="simple_white",
        xaxis_title="X Domain",
        yaxis_title="Expectation Value",
        yaxis_range=[-1, 1],
    )
    fig_hist.update_layout(
        title="Histogram (Absolute Value)",
        template="simple_white",
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
        yaxis_range=[0, 0.5],
    )
    return [fig_hist, fig_expval]


if __name__ == "__main__":
    app.run(debug=True)
