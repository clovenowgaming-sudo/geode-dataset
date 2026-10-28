#!/usr/bin/env python3
"""Kumpulkan docs + kode Geode SDK 5.11.2 dan kode mod Geode open-source jadi dataset training.

Mode penuh   : python build_dataset.py --out out --mods-minutes 150
Mode docs saja: python build_dataset.py --docs-only --out docs
"""
import argparse, gzip, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

TAG = "v5.11.2"
VER = "5.11.2"
ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0")
CODE_EXT = {".cpp", ".hpp", ".h", ".hh", ".cc", ".cxx", ".inl", ".mm", ".c", ".cmake", ".bro", ".md"}
NAMES = {"mod.json", "CMakeLists.txt"}
SKIP_DIRS = {".git", "build", "node_modules", "__pycache__", ".vs", ".idea", ".vscode"}
MAXF = 400_000
MAXF_BRO = 8_000_000
GH = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def clone(url, dest, branch=None, timeout=900):
    cmd = ["git", "clone", "--depth", "1", "-q"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, dest]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest, ignore_errors=True)
        return False
    if r.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
    return r.returncode == 0


def license_of(d):
    try:
        for fn in os.listdir(d):
            if fn.lower().startswith(("license", "licence", "copying")):
                t = open(os.path.join(d, fn), "rb").read(4000).decode("utf-8", "ignore").lower()
                for k, v in (("boost software", "BSL-1.0"), ("mit license", "MIT"),
                             ("apache license", "Apache-2.0"), ("gnu lesser", "LGPL"),
                             ("gnu general public", "GPL"), ("mozilla public", "MPL"),
                             ("unlicense", "Unlicense"), ("bsd", "BSD"), ("permission is hereby granted", "MIT")):
                    if k in t:
                        return v
                return "other"
    except Exception:
        pass
    return "none"


def wanted(rel, fn):
    ext = os.path.splitext(fn)[1].lower()
    if ext in CODE_EXT or fn in NAMES:
        return True
    return rel.startswith(".github/workflows/") and ext in (".yml", ".yaml")


def collect(repo_dir, repo, budget=25_000_000):
    out, total = [], 0
    lic = license_of(repo_dir)
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, repo_dir).replace(os.sep, "/")
            if not wanted(rel, fn):
                continue
            try:
                sz = os.path.getsize(p)
                limit = MAXF_BRO if fn.endswith(".bro") else MAXF
                if sz == 0 or sz > limit:
                    continue
                raw = open(p, "rb").read()
            except Exception:
                continue
            if b"\0" in raw:
                continue
            text = raw.decode("utf-8", "ignore")
            out.append({"repo": repo, "path": rel, "license": lic, "text": text})
            total += len(text)
            if total > budget:
                break
        if total > budget:
            break
    return out


def bundle_docs(docs_dir, geode_dir, out_path):
    n = 0
    skip = ("libs/", "extern/", "thirdparty/", "bindings/")
    with open(out_path, "w", encoding="utf-8") as o:
        o.write("# Geode SDK %s - dokumentasi lengkap\n# Sumber: geode-sdk/docs dan geode-sdk/geode @ %s\n" % (VER, TAG))
        for base, label in ((docs_dir, "docs"), (geode_dir, "geode")):
            if not base or not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for fn in sorted(files):
                    if not fn.lower().endswith(".md"):
                        continue
                    p = os.path.join(root, fn)
                    rel = os.path.relpath(p, base).replace(os.sep, "/")
                    if label == "geode" and any(s in rel + "/" for s in skip):
                        continue
                    try:
                        text = open(p, "rb").read().decode("utf-8", "ignore")
                    except Exception:
                        continue
                    o.write("\n\n<!-- FILE: %s/%s -->\n%s" % (label, rel, text))
                    n += 1
    return n


def bundle_headers(geode_dir, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8") as o:
        o.write("// Geode SDK %s - header API (loader/include) dan binding GD (.bro)\n" % VER)
        for root, dirs, files in os.walk(geode_dir):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for fn in sorted(files):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, geode_dir).replace(os.sep, "/")
                ext = os.path.splitext(fn)[1].lower()
                ok = (rel.startswith("loader/include/") and ext in (".hpp", ".h", ".hh", ".inl")) or ext == ".bro"
                if not ok:
                    continue
                try:
                    if os.path.getsize(p) > MAXF_BRO:
                        continue
                    text = open(p, "rb").read().decode("utf-8", "ignore")
                except Exception:
                    continue
                o.write("\n\n// ===== FILE: %s =====\n%s" % (rel, text))
                n += 1
    return n


def find_repos(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("repository", "source", "source_url", "github") and isinstance(v, str):
                m = GH.search(v)
                if m:
                    acc.add("%s/%s" % (m.group(1), m.group(2).removesuffix(".git")))
            else:
                find_repos(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_repos(v, acc)


def fetch_mod_repos(deadline):
    acc, page = set(), 1
    while page <= 400 and time.time() < deadline:
        url = "https://api.geode-sdk.org/v1/mods?page=%d&per_page=100" % page
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "geode-dataset"})
            data = json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            print("API berhenti di halaman %d: %s" % (page, e), flush=True)
            break
        pl = data.get("payload", data) if isinstance(data, dict) else data
        items = pl.get("data") if isinstance(pl, dict) else pl
        if not items:
            break
        find_repos(items, acc)
        page += 1
    print("Repo mod ditemukan:", len(acc), flush=True)
    return sorted(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--docs-only", action="store_true")
    ap.add_argument("--mods-minutes", type=float, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="geode-ds-")
    t0 = time.time()

    core = {}
    for name, branch in (("geode", TAG), ("docs", None), ("DevTools", None), ("NodeIDs", None), ("example-mod", None)):
        if a.docs_only and name not in ("geode", "docs"):
            continue
        d = os.path.join(work, "core", name)
        ok = clone("https://github.com/geode-sdk/" + name, d, branch)
        print("clone geode-sdk/%s: %s" % (name, "ok" if ok else "GAGAL"), flush=True)
        if ok:
            core[name] = d
            if name == "geode":
                subprocess.run(["git", "-C", d, "submodule", "update", "--init", "--depth", "1", "bindings"],
                               capture_output=True, text=True, timeout=900, env=ENV)

    n_docs = bundle_docs(core.get("docs"), core.get("geode"), os.path.join(a.out, "geode-docs-%s.md" % VER))
    n_hdr = bundle_headers(core["geode"], os.path.join(a.out, "geode-api-headers-%s.txt" % VER)) if "geode" in core else 0
    print("Docs: %d file md | Header/binding: %d file" % (n_docs, n_hdr), flush=True)
    if a.docs_only:
        shutil.rmtree(work, ignore_errors=True)
        return

    seen, stats = set(), {"files": 0, "bytes": 0, "repos_ok": 0, "repos_gagal": 0, "lisensi": {}}
    gz = gzip.open(os.path.join(a.out, "geode-code.jsonl.gz"), "wt", encoding="utf-8")

    def write(recs):
        for r in recs:
            h = hashlib.sha1(r["text"].encode("utf-8", "ignore")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            gz.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats["files"] += 1
            stats["bytes"] += len(r["text"])
            stats["lisensi"][r["license"]] = stats["lisensi"].get(r["license"], 0) + 1

    for name, d in core.items():
        write(collect(d, "geode-sdk/" + name, 120_000_000))
        stats["repos_ok"] += 1
    print("Core selesai: %d file" % stats["files"], flush=True)

    deadline = t0 + a.mods_minutes * 60
    slugs = [s for s in fetch_mod_repos(deadline) if not s.lower().startswith("geode-sdk/")]

    def do_mod(slug):
        if time.time() > deadline:
            return slug, None
        d = os.path.join(work, "mods", slug.replace("/", "__"))
        if not clone("https://github.com/" + slug, d, timeout=90):
            return slug, None
        recs = collect(d, slug)
        shutil.rmtree(d, ignore_errors=True)
        return slug, recs

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(do_mod, s) for s in slugs]
        for f in as_completed(futs):
            slug, recs = f.result()
            done += 1
            if recs is None:
                stats["repos_gagal"] += 1
            else:
                stats["repos_ok"] += 1
                write(recs)
            if done % 50 == 0:
                print("mod %d/%d | file %d | %.0f MB" % (done, len(slugs), stats["files"], stats["bytes"] / 1e6), flush=True)
    gz.close()
    stats["menit"] = round((time.time() - t0) / 60, 1)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
