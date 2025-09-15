# === save_artifacts.py ===
# Save the cluster IDs/confidences produced by run_clustering.py into ./clusters/ for use by defense script.
# ==== SAVE ARTIFACTS FOR ATTACK SCRIPT ====
import os, json
import numpy as np
from run_clustering import ids_final, conf_final  # from run_clustering.py
os.makedirs("./clusters", exist_ok=True)

dataset_name = "MNIST"  # <-- matches the key used in your attack script loop

# 1) required
np.save(f"./clusters/{dataset_name}_ids.npy", ids_final.astype(np.int64))
np.save(f"./clusters/{dataset_name}_conf.npy", conf_final.astype(np.float32))

# 2) optional pairs (you can skip; the attack script has a default pairing 0↔1,2↔3,...)
pairs = [[i, i+1] for i in range(0, 10, 2)]
with open(f"./clusters/{dataset_name}_pairs.json", "w") as f:
    json.dump(pairs, f)

print(f"[OK] Wrote ./clusters/{dataset_name}_ids.npy, _conf.npy, _pairs.json")