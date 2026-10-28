import os
HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "make_notebook.py")
s = open(p, encoding="utf-8").read()

# a) tambah paket gguf
a_old = "bitsandbytes datasets requests"
assert a_old in s
s = s.replace(a_old, "bitsandbytes datasets requests gguf", 1)

# b) model dasar = hasil konversi MicroMind.gguf
b_old = 'BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"'
assert b_old in s
s = s.replace(b_old, 'BASE = "micromind_hf"   # hasil konversi dari MicroMind.gguf buatan kita', 1)

# c) sel baru: unduh MicroMind.gguf dari release, ubah ke format HF (dekuantisasi Q8_0 -> fp16)
cell = """import subprocess, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

GGUF_URL = "https://github.com/clovenowgaming-sudo/micromind/releases/latest/download/MicroMind.gguf"
os.makedirs("gguf_in", exist_ok=True)
subprocess.run(["curl", "-L", "-s", "-o", "gguf_in/MicroMind.gguf", GGUF_URL], check=True)
print("MicroMind.gguf: %.2f GB" % (os.path.getsize("gguf_in/MicroMind.gguf") / 1e9))

m0 = AutoModelForCausalLM.from_pretrained("gguf_in", gguf_file="MicroMind.gguf", torch_dtype=torch.float16)
try:
    tok0 = AutoTokenizer.from_pretrained("gguf_in", gguf_file="MicroMind.gguf")
except Exception as e:
    print("Tokenizer GGUF gagal (%s), pakai tokenizer Qwen yang kosakatanya sama" % e)
    tok0 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
if not tok0.chat_template:
    tok0.chat_template = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct").chat_template
m0.save_pretrained(BASE)
tok0.save_pretrained(BASE)
del m0
print("MicroMind.gguf -> HF selesai:", sorted(os.listdir(BASE)))
"""
marker = "code(r'''SYS = "
assert s.count(marker) == 1
s = s.replace(marker, "code(r'''" + cell + "''')\n\n" + marker, 1)

open(p, "w", encoding="utf-8").write(s)
print("patched")
