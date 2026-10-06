import json

with open("Lab/kalman_fusion_lab_STUDENT.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source_str = "".join(cell["source"])

    # Fix make_F
    if "def make_F(dt):" in source_str and "NotImplementedError" in source_str:
        source_str = source_str.replace(
            "    F = np.eye(4)\n    # TODO: x should move by vx*dt and y by vy*dt. Which two entries of F do that?\n    raise NotImplementedError(\"Complete make_F\")",
            "    F = np.eye(4)\n    F[0, 2] = dt\n    F[1, 3] = dt\n    return F"
        )
    # Fix make_H
    if "def make_H():" in source_str and "NotImplementedError" in source_str:
        source_str = source_str.replace(
            "    H = np.zeros((2, 4))\n    # TODO: put the two 1s in the right places\n    raise NotImplementedError(\"Complete make_H\")",
            "    H = np.zeros((2, 4))\n    H[0, 0] = 1.0\n    H[1, 1] = 1.0\n    return H"
        )
    # Fix predict
    if "def predict(self, F, Q):" in source_str and "NotImplementedError" in source_str:
        source_str = source_str.replace(
            "        # TODO\n        raise NotImplementedError(\"Complete predict\")",
            "        self.x = F @ self.x\n        self.P = F @ self.P @ F.T + Q"
        )
    # Fix update
    if "def update(self, z, H, R):" in source_str and "NotImplementedError" in source_str:
        source_str = source_str.replace(
            "        # TODO: y = z - H x ; S = H P H^T + R ; K = P H^T S^-1 ; x <- x + K y ; P <- (I - K H) P\n        raise NotImplementedError(\"Complete update\")",
            "        y = z - H @ self.x\n        S = H @ self.P @ H.T + R\n        K = self.P @ H.T @ np.linalg.inv(S)\n        self.x = self.x + K @ y\n        I = np.eye(self.P.shape[0])\n        self.P = (I - K @ H) @ self.P\n        return y, S, K"
        )
    # Fix run_fusion
    if "def run_fusion(meas, q, x0, P0, t0=0.0):" in source_str and "NotImplementedError" in source_str:
        source_str = source_str.replace(
            "        # TODO 1: dt = ts - t_prev ; if dt > 0 -> kf.predict(make_F(dt), make_Q(dt, q))\n        # TODO 2: kf.update(z, H, R)           <- uses THIS sensor's H and R\n        # TODO 3: t_prev = ts ; log.append((ts, kf.x.copy(), kf.P.copy()))\n        raise NotImplementedError(\"Complete run_fusion\")",
            "        dt = ts - t_prev\n        if dt > 0:\n            kf.predict(make_F(dt), make_Q(dt, q))\n        kf.update(z, H, R)\n        t_prev = ts\n        log.append((ts, kf.x.copy(), kf.P.copy()))"
        )
    # Fix gated_update
    if "def gated_update(kf, z, H, R, p=0.99):" in source_str and "NotImplementedError" in source_str:
        source_str = source_str.replace(
            "    # TODO 1: y = z - H @ kf.x ;  S = H @ kf.P @ H.T + R\n    # TODO 2: d2 = y^T S^-1 y   (np.linalg.solve(S, y) is better than inv)\n    # TODO 3: if d2 > chi2.ppf(p, df=len(z)) -> return False ; else kf.update(z, H, R) and return True\n    raise NotImplementedError(\"Complete gated_update\")",
            "    y = z - H @ kf.x\n    S = H @ kf.P @ H.T + R\n    d2 = y.T @ np.linalg.solve(S, y)\n    if d2 > chi2.ppf(p, df=len(z)):\n        return False\n    else:\n        kf.update(z, H, R)\n        return True"
        )

    # Convert string back to list of strings with newlines for Jupyter format
    cell["source"] = [line + "\n" for line in source_str.split("\n")]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open("Lab/kalman_fusion_lab_STUDENT.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
