import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# GPR (Gaussian Process Regressor)

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


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.utils import algorithm_globals


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# =========================
# Učitavanje CSV
# =========================
csv_path = '/Users/milan/Desktop/GHQ/data/loto7_4510_k89.csv'
df = pd.read_csv(csv_path, header=0)

# =========================
# Uzimamo sve kombinacije
# =========================
N = len(df)
df = df.tail(N).reset_index(drop=True)

# =========================
# Priprema podataka
# =========================
X = df.iloc[:, :-1].values   # prvih 6 brojeva
y_full = df.values           # svih 7 brojeva (6+1)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# =========================
# Treniranje i predikcija po brojevima
# =========================
predicted_combination = []

for i in range(7):  # 6 brojeva + dodatni broj
    print(f"\n--- Treniranje QGPR modela za broj {i+1} ---")
    y = y_full[:, i]
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    kernel = DotProduct() + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, random_state=0)

    gpr.fit(X_scaled, y_scaled)


    # =========================
    # Predikcija sledeće kombinacije iz svih prethodnih
    # =========================
    all_input_scaled = np.mean(X_scaled, axis=0).reshape(1, -1)
    pred_scaled = gpr.predict(all_input_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).round().astype(int)[0][0]

    predicted_combination.append(pred)
    print(f"Predikcija za broj {i+1}: {pred}")

"""
--- Treniranje QGPR modela za broj 1 ---
Predikcija za broj 1: 5

--- Treniranje QGPR modela za broj 2 ---
Predikcija za broj 2: 10

--- Treniranje QGPR modela za broj 3 ---
Predikcija za broj 3: x

--- Treniranje QGPR modela za broj 4 ---
Predikcija za broj 4: x

--- Treniranje QGPR modela za broj 5 ---
Predikcija za broj 5: x

--- Treniranje QGPR modela za broj 6 ---
Predikcija za broj 6: 30

--- Treniranje QGPR modela za broj 7 ---
Predikcija za broj 7: 35
"""


# =========================
# Ispis rezultata
# =========================
predicted_combination = [int(x) for x in predicted_combination]
print("\n=== Predviđena sledeća loto kombinacija (7) ===")
print(" ".join(str(num) for num in predicted_combination))
print()


"""
=== Predviđena sledeća loto kombinacija (7) ===
5 10 x x x 30 35
"""
