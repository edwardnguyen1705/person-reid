import os
import shutil

ROOT = "saved"

list_id_removed = []


def rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path)


for folder_id in os.listdir(os.path.join(ROOT, "checkpoints")):
    if len(os.listdir(os.path.join(ROOT, "checkpoints", folder_id))) == 0:
        list_id_removed.append(folder_id)
    if not os.path.exists(os.path.join(ROOT, "logs", folder_id)):
        list_id_removed.append(folder_id)

for folder_id in os.listdir(os.path.join(ROOT, "logs")):
    if len(os.listdir(os.path.join(ROOT, "logs", folder_id))) == 0:
        list_id_removed.append(folder_id)
    if not os.path.exists(os.path.join(ROOT, "checkpoints", folder_id)):
        list_id_removed.append(folder_id)


for run_id in list_id_removed:
    rmtree(os.path.join(ROOT, "checkpoints", run_id))
    rmtree(os.path.join(ROOT, "logs", run_id))


# WARNING: Only run this when no have training process