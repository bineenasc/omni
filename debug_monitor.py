"""
Script de diagnostico - rode com: py -3.13 debug_monitor.py
"""
import os, glob, sys

HERE     = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(HERE, "controllers", "soccer_supervisor", "logs")

print(f"Python: {sys.version}")
print(f"HERE: {HERE}")
print(f"LOGS_DIR: {LOGS_DIR}")
print(f"LOGS_DIR existe? {os.path.isdir(LOGS_DIR)}")
print()

dirs = glob.glob(os.path.join(LOGS_DIR, "ppo_*"))
print(f"Diretorios encontrados: {dirs}")
print()

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    print("tensorboard importado OK")
except ImportError as e:
    print(f"ERRO ao importar tensorboard: {e}")
    sys.exit(1)

for d in sorted(dirs):
    print(f"\n--- {os.path.basename(d)} ---")
    files = glob.glob(os.path.join(d, "events.out.tfevents.*"))
    print(f"  tfevents files: {len(files)}")
    ea = EventAccumulator(d)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(f"  scalar tags: {tags}")
    for tag in tags:
        evts = ea.Scalars(tag)
        if evts:
            print(f"  {tag}: {len(evts)} pts, ultimo step={evts[-1].step}, val={evts[-1].value:.3f}")
