import time
from orchestrator import Orchestrator

t0 = time.time()
o = Orchestrator()
print(f"init (data load + graph build): {time.time()-t0:.1f}s")

q = "compare weekly packages of zong jazz and ufone by onnet minutes"
for run in (1, 2):
    print(f"--- run {run}")
    t0 = time.time(); n = 0
    for ev, p in o.run_stream(q):
        now = time.time() - t0
        if ev == "token":
            n += 1
            if n == 1:
                print(f"{now:6.1f}s  first token")
        else:
            print(f"{now:6.1f}s  {ev}")
    print(f"{time.time()-t0:6.1f}s  done ({n} chunks)")