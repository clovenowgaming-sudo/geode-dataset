import json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

C = []


def md(t):
    C.append(("markdown", t))


def code(t):
    C.append(("code", t))


md("# MicroMind-Geode: QLoRA kecil di Colab\n"
   "1. Runtime -> Change runtime type -> **T4 GPU**\n"
   "2. Colab Secrets (ikon kunci) -> tambah `GITHUB_TOKEN` (token GitHub dengan scope `repo`), aktifkan akses notebook\n"
   "3. Ubah `RUN`, `MAX_SAMPLES`, `ONLY_PERMISSIVE` di sel konfigurasi tiap putaran, lalu Run all")

code("!pip install -q -U transformers peft accelerate bitsandbytes datasets requests gguf")

code(r'''import os, json, gzip, random, requests
from google.colab import userdata

REPO = "clovenowgaming-sudo/geode-dataset"
BASE = "micromind_hf"   # hasil konversi dari MicroMind.gguf buatan kita
RUN = "v1"                 # ganti tiap putaran: v2, v3, ...
ONLY_PERMISSIVE = False    # True = hanya lisensi MIT/BSL/Apache/Unlicense (untuk versi publik)
MAX_SAMPLES = 3000         # naikkan bertahap
EPOCHS = 1
MAX_LEN = 1024

H = {"Authorization": "Bearer " + userdata.get("GITHUB_TOKEN"), "Accept": "application/vnd.github+json"}
rel = requests.get("https://api.github.com/repos/%s/releases/latest" % REPO, headers=H).json()
os.makedirs("data", exist_ok=True)
for a in rel["assets"]:
    r = requests.get(a["url"], headers={**H, "Accept": "application/octet-stream"})
    open("data/" + a["name"], "wb").write(r.content)
    print(a["name"], len(r.content))
''')

code(r'''import subprocess, torch
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
''')

code(r'''SYS = "Kamu adalah MicroMind, asisten AI untuk coding dan pengembangan aplikasi. Tulis kode lengkap yang bisa langsung dijalankan, jelaskan singkat, dan jawab dengan bahasa yang dipakai pengguna."
OK_LIC = {"BSL-1.0", "MIT", "Apache-2.0", "Unlicense"}
random.seed(0)
rows = [json.loads(l) for l in gzip.open("data/geode-code.jsonl.gz", "rt", encoding="utf-8")]
if ONLY_PERMISSIVE:
    rows = [r for r in rows if r["license"] in OK_LIC]
rows = [r for r in rows if 80 < len(r["text"]) < 6000 and not r["path"].endswith(".bro")]
random.shuffle(rows)

def ask(r):
    p, repo = r["path"], r["repo"]
    if p.endswith("mod.json"):
        return "Tulis mod.json untuk mod Geode (Geode SDK 5.11.2) seperti di repo %s." % repo
    if p.endswith("CMakeLists.txt"):
        return "Tulis CMakeLists.txt untuk mod Geode seperti di repo %s." % repo
    if p.endswith(".md"):
        return "Tulis dokumen %s untuk mod Geode %s." % (p, repo)
    return "Tulis file %s untuk mod Geode %s (Geode SDK 5.11.2)." % (p, repo)

samples = [(ask(r), r["text"]) for r in rows[:MAX_SAMPLES]]
docs = open("data/geode-docs-5.11.2.md", encoding="utf-8").read().split("<!-- FILE: ")[1:]
for d in docs:
    name, _, body = d.partition(" -->")
    body = body.strip()
    if len(body) > 200:
        samples.append(("Jelaskan dokumentasi Geode SDK 5.11.2: %s" % name, body[:6000]))
print(len(samples), "sampel")
''')

code(r'''import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

tok = AutoTokenizer.from_pretrained(BASE)

def fmt(q, a):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return tok.apply_chat_template(msgs, tokenize=False)

ds = Dataset.from_dict({"text": [fmt(q, a) for q, a in samples]})
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="out", per_device_train_batch_size=2, gradient_accumulation_steps=8,
    num_train_epochs=EPOCHS, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    fp16=True, logging_steps=10, save_strategy="no", report_to="none",
    optim="paged_adamw_8bit", gradient_checkpointing=True)
Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
model.save_pretrained("lora_out")
''')

code(r'''EVAL = [
    ("Buat hook untuk MenuLayer::init di Geode yang menulis log.", ["$modify", "MenuLayer", "init", "log::"]),
    ("Tulis mod.json untuk mod id user.halo versi v1.0.0, geode 5.11.2, gd 2.2081.", ['"id"', '"version"', '"geode"', '"gd"', "user.halo", "5.11.2"]),
    ("Bagaimana membaca setting bool bernama aktif di mod Geode?", ["Mod::get()", "getSettingValue", "bool"]),
    ("Buat tombol di MenuLayer yang membuka FLAlertLayer di Geode.", ["CCMenuItemSpriteExtra", "FLAlertLayer", "menu_selector", "$modify"]),
    ("Hook PlayLayer::update di Geode.", ["$modify", "PlayLayer", "update", "float"]),
    ("Tulis CMakeLists.txt standar untuk mod Geode.", ["find_package(Geode", "setup_geode_mod", "add_library"]),
    ("Cara menyimpan dan membaca data dengan Mod::get()->setSavedValue di Geode.", ["setSavedValue", "getSavedValue", "Mod::get()"]),
    ("Jalankan kode saat mod Geode selesai dimuat memakai $on_mod.", ["$on_mod", "Loaded"]),
    ("Hook GJBaseGameLayer::processCommands di Geode.", ["$modify", "GJBaseGameLayer", "processCommands"]),
    ("Tampilkan Notification toast di Geode.", ["Notification::create", "show()"]),
    ("Hook LevelInfoLayer::init dan tambahkan CCLabelBMFont.", ["$modify", "LevelInfoLayer", "CCLabelBMFont", "addChild"]),
    ("Tulis setting integer dengan min dan max di mod.json Geode.", ['"settings"', '"type"', '"int"', '"min"', '"max"']),
    ("Buat popup custom dengan geode::Popup.", ["Popup", "setup", "create"]),
    ("Contoh log info, warn, dan error di Geode.", ["log::info", "log::warn", "log::error"]),
    ("Hook CCScheduler::update untuk mengubah dt di Geode.", ["$modify", "CCScheduler", "update"]),
    ("Cara mendapatkan pointer PlayLayer yang sedang aktif di Geode.", ["PlayLayer::get()"]),
    ("Workflow GitHub Actions untuk build mod Geode semua platform.", ["geode-sdk/build-geode-mod", "runs-on", "uses:"]),
    ("Jelaskan $modify dan $override di Geode dengan contoh.", ["$modify", "$override"]),
    ("Ambil ukuran layar dengan cocos2d di mod Geode.", ["CCDirector::get()", "getWinSize"]),
    ("Hook PauseLayer::customSetup di Geode.", ["$modify", "PauseLayer", "customSetup"]),
]

def score(text, kws):
    return sum(k in text for k in kws) / len(kws)

@torch.no_grad()
def gen(q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=300, do_sample=False, use_cache=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

model.eval()
tb = tt = 0
for q, kws in EVAL:
    with model.disable_adapter():
        b = score(gen(q), kws)
    t = score(gen(q), kws)
    tb += b; tt += t
    print("%.2f -> %.2f | %s" % (b, t, q[:60]))
print("RATA-RATA  sebelum: %.3f  sesudah: %.3f" % (tb / len(EVAL), tt / len(EVAL)))
''')

code(r'''import gc
from google.colab import drive
from peft import PeftModel

del model
gc.collect(); torch.cuda.empty_cache()
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
m = PeftModel.from_pretrained(base, "lora_out").merge_and_unload()
m.save_pretrained("merged"); tok.save_pretrained("merged")
OLD = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
for fn in os.listdir("merged"):
    if fn.endswith((".json", ".jinja")):
        p = os.path.join("merged", fn)
        s = open(p, encoding="utf-8").read()
        if OLD in s:
            open(p, "w", encoding="utf-8").write(s.replace(OLD, SYS))
drive.mount("/content/drive")
''')

code("!git clone --depth 1 https://github.com/ggml-org/llama.cpp\n"
     "!pip install -q ./llama.cpp/gguf-py sentencepiece\n"
     "!python llama.cpp/convert_hf_to_gguf.py merged --outtype q8_0 --model-name MicroMind-Geode --outfile MicroMind-Geode-{RUN}.gguf\n"
     "!cp MicroMind-Geode-{RUN}.gguf /content/drive/MyDrive/\n"
     "!ls -lh /content/drive/MyDrive/*.gguf")

cells = []
for kind, text in C:
    cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cells.append(cell)

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("colab_train.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("colab_train.ipynb dibuat:", len(cells), "sel")

for cmd in ("git add -A", 'git commit -m "Notebook Colab QLoRA + eval Geode" || true', "git push"):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    print("$", cmd, "\n", (r.stdout + r.stderr).strip())
