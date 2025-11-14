import warnings
warnings.filterwarnings("ignore", message="No gradient function provided")

"""
QGPR (Quantum Gaussian Process Regressor)
"""

"""
| Paket                       | Verzija |
| --------------------------- | ------- |
| **python**                  | 3.11.13 |
| **qiskit**                  | 1.4.4   |
| **qiskit-machine-learning** | 0.8.3   |
| **qiskit-ibm-runtime**      | 0.43.0  |
| **macOS**                   | Tahos   |
| **Apple**                   | M1      |
"""

"""
https://github.com/forsing
https://github.com/forsing?tab=repositories
"""

"""
Loto Skraceni Sistemi
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""

"""
svih 4510 izvlacenja
30.07.1985.- 11.11.2025.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.visualization import circuit_drawer

from IPython.display import display
from IPython.display import clear_output

from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# 1. Učitaj loto podatke
df = pd.read_csv("/Users/milan/Desktop/GHQ/data/loto7_4510_k89.csv", header=None)


###################################
print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""
Prvih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4   5   6
0   5  14  15  17  28  30  34
1   2   3  13  18  19  23  37
2  13  17  18  20  21  26  39
3  17  20  23  26  35  36  38
4   3   4   8  11  29  32  37
"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""
Zadnjih 5 ucitanih kombinacija iz CSV fajla:

       0   1   2   3   4   5   6
4505   6   9  12  20  24  31  36
4506  10  14  17  24  32  35  36
4507   5   9  10  13  16  28  34
4508  10  13  14  21  30  32  39
4509   2   7  11  23  26  38  39
"""
####################################


# 2. Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]

# 3. Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# 4. Primeni mapiranje
df_indexed = map_to_indexed_range(df, min_val, max_val)

# 5. Provera rezultata
print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 4510, Broj pozicija: 7
"""


print()
print("Prvih 5 mapiranih kombinacija:")
print()
print(df_indexed.head())
print()
"""
Prvih 5 mapiranih kombinacija:

    0   1   2   3   4   5   6
0   4  12  12  13  23  24  27
1   1   1  10  14  14  17  30
2  12  15  15  16  16  20  32
3  16  18  20  22  30  30  31
4   2   2   5   7  24  26  30
"""

print()
print("Zadnjih 5 mapiranih kombinacija:")
print()
print(df_indexed.tail())
print()
"""
Zadnjih 5 mapiranih kombinacija:

      0   1   2   3   4   5   6
4505  5   7   9  16  19  25  29
4506  9  12  14  20  27  29  29
4507  4   7   7   9  11  22  27
4508  9  11  11  17  25  26  32
4509  1   5   8  19  21  32  32
"""




# Parametri
num_qubits = 5          # 5 qubita po poziciji
num_layers = 2          # Dubina varijacionog sloja
num_positions = 5       # 7 pozicija (brojeva) u loto kombinaciji

# Enkoder: binarno kodiranje vrednosti u kvantni registar
def encode_position(value):
    """
    Sigurno enkoduje 'value' u QuantumCircuit sa tacno num_qubits qubita.
    Ako value zahteva vise bitova od num_qubits, koristi se LSB (zadnjih num_qubits bitova),
    i ispisuje se upozorenje.
    """
    # osiguraj int
    v = int(value)
    bin_full = format(v, 'b')  # pravi binarni bez vodećih nula
    if len(bin_full) > num_qubits:
        # upozorenje: vrednost ne staje u broj qubita; koristimo zadnjih num_qubits bita (LSB)
        print(f"Upozorenje: value={v} zahteva {len(bin_full)} bitova, a num_qubits={num_qubits}. Koristim zadnjih {num_qubits} bita.")
        bin_repr = bin_full[-num_qubits:]
    else:
        bin_repr = bin_full.zfill(num_qubits)

    qc = QuantumCircuit(num_qubits)
    # reversed da bi LSB išao na qubit 0 (ako želiš suprotno, ukloni reversed)
    for i, bit in enumerate(reversed(bin_repr)):
        if bit == '1':
            qc.x(i)
    return qc


# Varijacioni sloj: Ry rotacije + CNOT lanac
def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return qc

# QCBM ansambl: slojevi varijacionih blokova
def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer * num_qubits
        end = (layer + 1) * num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc

# Kompletan QCBM za svih 7 pozicija
def full_qcbm(params_list, values):
    total_qubits = num_qubits * num_positions
    qc = QuantumCircuit(total_qubits)

    for pos in range(num_positions):
        start_q = pos * num_qubits
        end_q = start_q + num_qubits

        # Enkoduj vrednost za poziciju
        qc_enc = encode_position(values[pos])
        qc.compose(qc_enc, qubits=range(start_q, end_q), inplace=True)

        # Dodaj varijacioni ansambl
        qc_var = qcbm_ansatz(params_list[pos])
        qc.compose(qc_var, qubits=range(start_q, end_q), inplace=True)

    # Dodaj merenja za svih 35 qubita
    qc.measure_all()

    return qc

# Test primer: enkoduj kombinaciju [13, 5, 7, 20, 23, 8, 15]
test_values = [12, 11, 38, 14, 35, 4, 21]
np.random.seed(39)
params_list = [np.random.uniform(0, 2*np.pi, num_layers * num_qubits) for _ in range(num_positions)]

# Generiši QCBM za svih 7 pozicija
full_circuit = full_qcbm(params_list, test_values)



# Prikaz celog kruga u 'mpl' formatu
full_circuit.draw('mpl')
# plt.show()

# fold=40 prelama linije tako da veliki krug stane na ekran.
full_circuit.draw('mpl', fold=40)
# plt.show()


# Kompaktni prikaz kola
print("\nKompaktni prikaz kvantnog kola (text):\n")
# print(full_circuit.draw('text'))
"""
Kompaktni prikaz kvantnog kola (text):
"""


# display(full_circuit.draw())     
display(full_circuit.draw("mpl"))
# plt.show()


circuit_drawer(full_circuit, output='latex', style={"backgroundcolor": "#EEEEEE"})
# plt.show()


# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/KvantniRegresor/6QGPR/QGPR_qc25_7_4.jpg")



####################



# ============================================================
# QGPR – PREDIKCIJA SLEDEĆE LOTO 7/39 KOMBINACIJE
# ============================================================

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

print("\n=====================================================")
print("  QGPR – Kvantna predikcija sledeće loto kombinacije")
print("=====================================================\n")

# Koristimo poslednjih 100 izvlačenja (po želji stavi 4510)
N = 100 
# N = 4510

X = df_indexed.iloc[-N:-1].values
y = df_indexed.iloc[-N+1:].values   # pomak za 1

# Quantum kernel
quantum_kernel = FidelityQuantumKernel()

models = []
predicted_indexed = []

# ------------------------------------------------------------
# Trenira se 7 QGPR modela – po jedan za svaku poziciju
# ------------------------------------------------------------
for pos in range(7):
    print(f"Trening QGPR modela za poziciju {pos+1}/7 ...")

    y_pos = y[:, pos]

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

    # kvantni kernel matrica
    K_train = quantum_kernel.evaluate(X, X)

    # treniraj GPR nad kvantnim kernelom
    gpr.fit(K_train, y_pos)

    models.append(gpr)

# ------------------------------------------------------------
# Predikcija sledeće kombinacije
# ------------------------------------------------------------
last_input = df_indexed.iloc[-1:].values

predicted_indexed = []
for pos in range(7):
    K_pred = quantum_kernel.evaluate(last_input, X)
    pred_val = models[pos].predict(K_pred)[0]

    # zaokruži i ograniči na validan opseg
    p = int(np.clip(round(pred_val), 0, max_val[pos] - min_val[pos]))
    predicted_indexed.append(p)

# vraćanje u originalni loto domen
predicted_final = [predicted_indexed[i] + min_val[i] for i in range(7)]


print("\n========================================")
print("     QGPR – PREDIKCIJA SLEDEĆIH 7 BROJEVA")
print("========================================\n")

print("Mapirani brojevi (0-range):")
print(predicted_indexed)

print("\nFinalni loto brojevi (1–39):")
print(predicted_final)

print("\n========================================\n")


"""
========================================
     QGPR – PREDIKCIJA SLEDEĆIH 7 BROJEVA
========================================

Mapirani brojevi (0-range):
[0, 0, x, x, x, 0, 4]

Finalni loto brojevi (1–39):
[1, 2, x, x, x, 6, 11]

N = 100 zadnjih kombinacija

========================================
"""
