import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_notebook.py")
s = open(p, encoding="utf-8").read()
old = '.replace("{RUN}", "$RUN").replace("$RUN", "v1"))'
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, ")"))
print("patched")
