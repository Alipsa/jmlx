import subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
stop=p/'stop-requested'
assert not stop.exists(), 'Stale stop request; inspect window state'
try:
    subprocess.run([sys.executable,'-I',str(p/'admin-window.py'),'connected'],check=True)
    print('Connected probe retained. Leave this Terminal open; waiting for verification to finish.',flush=True)
    deadline=time.monotonic()+1800
    while not stop.exists() and time.monotonic()<deadline:
        time.sleep(2)
finally:
    subprocess.run([sys.executable,'-I',str(p/'admin-window.py'),'stop'],check=True)
