import json

with open("Lab/kalman_fusion_lab_STUDENT.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    
    if "def make_F(dt):" in source and "NotImplementedError" in source:
        cell["source"] = [
            "def make_F(dt):\n",
            "    \"\"\"Build the constant-velocity state transition matrix.\n",
            "\n",
            "    Args:\n",
            "        dt: Time step, in seconds.\n",
            "\n",
            "    Returns:\n",
            "        numpy.ndarray: 4x4 matrix `F` such that `F @ [x, y, vx, vy]`\n",
            "        advances position by velocity * dt and leaves velocity unchanged.\n",
            "    \"\"\"\n",
            "    F = np.eye(4)\n",
            "    F[0, 2] = dt\n",
            "    F[1, 3] = dt\n",
            "    return F\n",
            "\n",
            "def make_H():\n",
            "    \"\"\"Build the measurement matrix for a position-only sensor.\n",
            "\n",
            "    Returns:\n",
            "        numpy.ndarray: 2x4 matrix `H` that extracts `[x, y]` from the\n",
            "        4-element state `[x, y, vx, vy]`.\n",
            "    \"\"\"\n",
            "    H = np.zeros((2, 4))\n",
            "    H[0, 0] = 1.0\n",
            "    H[1, 1] = 1.0\n",
            "    return H\n"
        ]
        
    elif "def predict(self, F, Q):" in source and "NotImplementedError" in source:
        cell["source"] = [
            "    def __init__(self, x0, P0):\n",
            "        \"\"\"Initialize the filter with a starting state and covariance.\n",
            "\n",
            "        Args:\n",
            "            x0: Initial state estimate.\n",
            "            P0: Initial state covariance.\n",
            "        \"\"\"\n",
            "        self.x = np.array(x0, dtype=float)      # state mean\n",
            "        self.P = np.array(P0, dtype=float)      # state covariance\n",
            "\n",
            "    def predict(self, F, Q):\n",
            "        \"\"\"Advance the state estimate through the process model.\n",
            "\n",
            "        Args:\n",
            "            F: State transition matrix for this time step.\n",
            "            Q: Process noise covariance for this time step.\n",
            "\n",
            "        Returns:\n",
            "            None: Updates `self.x` and `self.P` in place.\n",
            "        \"\"\"\n",
            "        self.x = F @ self.x\n",
            "        self.P = F @ self.P @ F.T + Q\n",
            "\n",
            "    def update(self, z, H, R):\n",
            "        \"\"\"Correct the state estimate using a new measurement.\n",
            "\n",
            "        Args:\n",
            "            z: Measurement vector.\n",
            "            H: Measurement matrix mapping state to measurement space.\n",
            "            R: Measurement noise covariance.\n",
            "\n",
            "        Returns:\n",
            "            tuple: `(y, S, K)` -- the innovation, innovation covariance, and\n",
            "            Kalman gain used for this update. `self.x` and `self.P` are\n",
            "            updated in place.\n",
            "        \"\"\"\n",
            "        y = z - H @ self.x\n",
            "        S = H @ self.P @ H.T + R\n",
            "        K = self.P @ H.T @ np.linalg.inv(S)\n",
            "        self.x = self.x + K @ y\n",
            "        I = np.eye(self.P.shape[0])\n",
            "        self.P = (I - K @ H) @ self.P\n",
            "        return y, S, K\n"
        ]

    elif "def run_fusion(meas, q, x0, P0, t0=0.0):" in source and "NotImplementedError" in source:
        cell["source"] = [
            "def run_fusion(meas, q, x0, P0, t0=0.0):\n",
            "    \"\"\"Run a Kalman filter over a mixed, asynchronous multi-sensor stream.\n",
            "\n",
            "    Args:\n",
            "        meas: List of `(timestamp, name, z, H, R)` tuples, sorted by time\n",
            "            (as returned by `make_measurements`).\n",
            "        q: Process noise strength passed to `make_Q`.\n",
            "        x0: Initial state estimate.\n",
            "        P0: Initial state covariance.\n",
            "        t0: Time of the initial state estimate.\n",
            "\n",
            "    Returns:\n",
            "        list: `(timestamp, x, P)` tuples logging the filter's state and\n",
            "        covariance immediately after each measurement update.\n",
            "    \"\"\"\n",
            "    kf = KalmanFilter(x0, P0)\n",
            "    t_prev = t0\n",
            "    log = []\n",
            "    for ts, name, z, H, R in meas:\n",
            "        dt = ts - t_prev\n",
            "        if dt > 0:\n",
            "            kf.predict(make_F(dt), make_Q(dt, q))\n",
            "        kf.update(z, H, R)\n",
            "        t_prev = ts\n",
            "        log.append((ts, kf.x.copy(), kf.P.copy()))\n",
            "    return log\n"
        ]

    elif "def gated_update(kf, z, H, R, p=0.99):" in source and "NotImplementedError" in source:
        cell["source"] = [
            "def gated_update(kf, z, H, R, p=0.99):\n",
            "    \"\"\"Apply a chi-squared gated Kalman update, rejecting outliers.\n",
            "\n",
            "    Args:\n",
            "        kf: A `KalmanFilter` instance, updated in place if the measurement\n",
            "            is accepted.\n",
            "        z: Measurement vector.\n",
            "        H: Measurement matrix.\n",
            "        R: Measurement noise covariance.\n",
            "        p: Gate probability; a measurement is rejected if its squared\n",
            "            Mahalanobis distance exceeds the chi-squared `p`-quantile.\n",
            "\n",
            "    Returns:\n",
            "        bool: True if the measurement was accepted and the filter updated,\n",
            "        False if it was rejected as an outlier (filter left unchanged).\n",
            "    \"\"\"\n",
            "    y = z - H @ kf.x\n",
            "    S = H @ kf.P @ H.T + R\n",
            "    d2 = y.T @ np.linalg.solve(S, y)\n",
            "    if d2 > chi2.ppf(p, df=len(z)):\n",
            "        return False\n",
            "    else:\n",
            "        kf.update(z, H, R)\n",
            "        return True\n"
        ]

with open("Lab/kalman_fusion_lab_STUDENT.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
