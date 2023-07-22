# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Diagonal matrix circuit."""

from __future__ import annotations
import cmath
import numpy as np
import sympy

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code

_EPS = 1e-10

import pprint


class DiagonalNew(QuantumCircuit):
    r"""Diagonal circuit.

    Circuit symbol:

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1 Diagonal ├
             │           │
        q_2: ┤2          ├
             └───────────┘

    Matrix form:

    .. math::
        \text{DiagonalGate}\ q_0, q_1, .., q_{n-1} =
            \begin{pmatrix}
                D[0]    & 0         & \dots     & 0 \\
                0       & D[1]      & \dots     & 0 \\
                \vdots  & \vdots    & \ddots    & 0 \\
                0       & 0         & \dots     & D[n-1]
            \end{pmatrix}

    Diagonal gates are useful as representations of Boolean functions,
    as they can map from {0,1}^2**n to {0,1}^2**n space. For example a phase
    oracle can be seen as a diagonal gate with {+1, -1} on the diagonals. Such
    an oracle will induce a +1 or -1 phase on the amplitude of any corresponding
    basis state.

    Diagonal gates appear in many classically hard oracular problems such as
    Forrelation or Hidden Shift circuits.

    Diagonal gates are represented and simulated more efficiently than a dense
    2**n x 2**n unitary matrix.

    The reference implementation is via the method described in
    Theorem 7 of [1]. The code is based on Emanuel Malvetti's semester thesis
    at ETH in 2018, supervised by Raban Iten and Prof. Renato Renner.

    **Reference:**

    [1] Shende et al., Synthesis of Quantum Logic Circuits, 2009
    `arXiv:0406176 <https://arxiv.org/pdf/quant-ph/0406176.pdf>`_
    """

    def __init__(self, diag: list[complex] | np.ndarray) -> None:
        """Create a new Diagonal circuit.

        Args:
            diag: list of the 2^k diagonal entries (for a diagonal gate on k qubits).

        Raises:
            CircuitError: if the list of the diagonal entries or the qubit list is in bad format;
                if the number of diagonal entries is not 2^k, where k denotes the number of qubits
        """
        if not isinstance(diag, (list, np.ndarray)):
            raise CircuitError("Diagonal entries must be in a list or numpy array.")
        num_qubits = np.log2(len(diag))
        if num_qubits < 1 or not num_qubits.is_integer():
            raise CircuitError("The number of diagonal entries is not a positive power of 2.")
        # if not np.allclose(np.abs(diag), 1, atol=_EPS):
        #     raise CircuitError("A diagonal element does not have absolute value one.")

        num_qubits = int(num_qubits)

        gate_list = [[] for _ in range(2**num_qubits)]

        # Since the diagonal is a unitary, all its entries have absolute value
        # one and the diagonal is fully specified by the phases of its entries.
        diag_phases = [cmath.phase(z) for z in diag]
        # angles_rz = np.array(sympy.fwht(diag_phases)).astype(float) / np.sqrt(2 ** (num_qubits - 2))
        from qiskit.circuit import ParameterVector

        angles_rz = ParameterVector("beta", 2 ** (num_qubits))

        for i in range(1, num_qubits):
            gate_list[0].append(["rz", -angles_rz[2 ** (num_qubits - i)], i - 1])

        cc_set = [0]
        gray_code = [0, 1]
        for p in range(2, num_qubits + 1):
            t = 2 ** (p - 1)
            cc_set[t // 2 - 1] = p - 1
            cc_set.extend(cc_set)
            if p < num_qubits:
                gate_list[2**p].append(["cx", 0, p - 1])
                for i in range(2, t + 1):
                    j = ((gray_code[i - 1] << 1) + 1) << (num_qubits - p)
                    gate_list[2**p + 2 * i - 3].append(["rz", -angles_rz[j], p - 1])
                    gate_list[2**p + 2 * i - 2].append(["cx", cc_set[i - 1] - 1, p - 1])
                gray_code = [x << 1 for x in gray_code] + [(x << 1) + 1 for x in gray_code[::-1]]

        for i in range(1, 2 ** (num_qubits - 1) + 1):
            j = (gray_code[i - 1] << 1) + 1
            gate_list[2 * i - 2].append(["rz", -angles_rz[j], num_qubits - 1])
            gate_list[2 * i - 1].append(["cx", cc_set[i - 1] - 1, num_qubits - 1])

        circuit = QuantumCircuit(num_qubits, name="Diagonal")
        circuit.global_phase += diag_phases[0]

        for seq in gate_list:
            for gate in seq:
                if gate[0] == "rz":
                    circuit.rz(gate[1], gate[2])
                elif gate[0] == "cx":
                    circuit.cx(gate[1], gate[2])

        super().__init__(num_qubits, name="Diagonal")
        self.append(circuit.to_gate(), self.qubits)
