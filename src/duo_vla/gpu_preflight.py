"""Non-mutating, fail-closed availability guard for explicitly selected GPUs.

No torch import or CUDA context. This host's A6000 devices are not authorized.
The guard never waits for a busy GPU, chooses another GPU, or kills its owner.
"""

from __future__ import annotations

import csv
import io
import subprocess


def require_idle_training_gpus(indices: tuple[int, ...]) -> dict:
    if indices not in ((0,), (0, 1)) or any(type(index) is not int for index in indices):
        raise ValueError("CALVIN is restricted to physical GPU 0 or 0,1; A6000 use is prohibited")
    selected = ",".join(map(str, indices))

    def query(fields, kind):
        result = subprocess.run(
            ["/usr/bin/nvidia-smi", f"--id={selected}", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return [[field.strip() for field in row] for row in csv.reader(io.StringIO(result.stdout)) if row]

    devices = query("index,uuid,name,memory.total", "gpu")
    if len(devices) != len(indices) or any(len(row) != 4 for row in devices):
        raise RuntimeError("cannot authenticate selected GPU inventory")
    if [int(row[0]) for row in devices] != list(indices):
        raise RuntimeError("nvidia-smi returned a different physical GPU selection")
    if any("A6000" in row[2].upper() for row in devices):
        raise RuntimeError("A6000 use is prohibited")
    if any(float(row[3]) < 80 * 1024 for row in devices):
        raise RuntimeError("CALVIN full-model TP1 optimization requires at least 80 GiB per selected GPU")
    processes = query("gpu_uuid,pid", "compute-apps")
    if any(len(row) != 2 for row in processes):
        raise RuntimeError("cannot authenticate GPU process inventory")
    selected_uuids = {row[1] for row in devices}
    busy = [row for row in processes if row[0] in selected_uuids]
    if busy:
        raise RuntimeError(f"selected GPUs are busy; refusing to start or interrupt existing owners: {busy}")
    return {"physical_gpu_indices": list(indices), "gpu_uuids": [row[1] for row in devices], "idle": True}
