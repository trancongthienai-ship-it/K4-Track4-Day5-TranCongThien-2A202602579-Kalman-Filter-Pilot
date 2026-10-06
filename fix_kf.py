import json

with open("Lab/kalman_fusion_lab_STUDENT.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    if len(source) > 0 and source[0].startswith("    def __init__(self, x0, P0):"):
        cell["source"] = [
            "class KalmanFilter:\n",
            "    \"\"\"A linear Kalman filter for state estimation.\n",
            "\n",
            "    Attributes:\n",
            "        x: Current state estimate (1-D numpy array).\n",
            "        P: Current state estimate covariance (2-D numpy array).\n",
            "    \"\"\"\n"
        ] + source

with open("Lab/kalman_fusion_lab_STUDENT.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
