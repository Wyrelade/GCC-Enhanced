#!/usr/bin/env python3
"""GCC-Enhanced Phase C batch driver.

Runs the solve.py pipeline over a list of (func, C-stub, retty) jobs and reports
VERIFIED / bytes-only / not-solved / error for each, caching the verified wins in a
manifest so a solved function is never re-searched.

A job source is either:
  * a JSON file: a list of {"func": ..., "cfile": ..., "retty": "void"|"int"} objects, or
  * a directory of stubs named f<FUNC>.c (retty defaults to void; override per-func in an
    optional sidecar <dir>/retty.json = {"func_XXXX": "int", ...}).

Only status == 'verified' is written to the manifest. 'bytes-only' means the produced
bytes equal the target but the equiv gate did NOT prove equivalence -- that is a suspect,
never a claimed match, and is surfaced loudly for review.

Usage:
    batch.py <jobs.json | stubs-dir> [--manifest PATH] [-v]
"""
import os, re, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asmlib
import solve as S

DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "solved_manifest.json")

def load_jobs(src):
    if os.path.isdir(src):
        sidecar = {}
        sc = os.path.join(src, "retty.json")
        if os.path.exists(sc):
            sidecar = json.load(open(sc, encoding="utf-8"))
        jobs = []
        for fn in sorted(os.listdir(src)):
            m = re.fullmatch(r"(func_[0-9A-Fa-f]+)\.c", fn)
            if not m:
                m = re.fullmatch(r"f([0-9A-Fa-f]+)\.c", fn)
                func = "func_" + m.group(1) if m else None
            else:
                func = m.group(1)
            if func:
                jobs.append({"func": func, "cfile": os.path.join(src, fn),
                             "retty": sidecar.get(func, "void")})
        return jobs
    return json.load(open(src, encoding="utf-8"))

def load_manifest(path):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {}

def save_manifest(path, man):
    json.dump(man, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)

TAG = {"verified": "VERIFIED  ", "bytes-only": "BYTES-ONLY", "already": "ALREADY   ",
       "not-solved": "not-solved", "error": "ERROR     "}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="jobs.json or a directory of f<FUNC>.c stubs")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    jobs = load_jobs(args.src)
    man = load_manifest(args.manifest)
    counts = {k: 0 for k in TAG}
    print("== GCC-Enhanced batch: %d job(s) ==\n" % len(jobs))
    for j in jobs:
        func, cfile, retty = j["func"], j["cfile"], j.get("retty", "void")
        if func in man and man[func].get("status") == "verified":
            print("%s %s  (cached)" % (TAG["verified"], func))
            counts["verified"] += 1
            continue
        r = S.solve_one(cfile, func, retty)
        st = r["status"]
        counts[st] = counts.get(st, 0) + 1
        extra = ""
        if st in ("verified", "bytes-only"):
            extra = "  sigma=%s" % {("$%s" % k): ("$%s" % v)
                                    for k, v in sorted(r["sigma"].items())}
        if st == "bytes-only":
            extra += "  [equiv NOT proven: %s]" % r.get("equiv_reason", "")
        if st == "error":
            extra = "  %s" % (r["error"] or "").splitlines()[0][:80]
        print("%s %s  (%s)%s" % (TAG.get(st, st), func, retty, extra))
        if args.verbose and st in ("bytes-only", "not-solved"):
            S.show(r["rows"], limit=40)
        if st == "verified":
            man[func] = {"status": "verified", "cfile": cfile, "retty": retty,
                         "sigma": {str(k): str(v) for k, v in r["sigma"].items()}}

    save_manifest(args.manifest, man)
    print("\n== summary ==")
    for k in ("verified", "already", "bytes-only", "not-solved", "error"):
        if counts.get(k):
            print("  %-11s %d" % (k, counts[k]))
    print("manifest:", args.manifest, "(%d verified)"
          % sum(1 for v in man.values() if v.get("status") == "verified"))
    # nonzero exit if any suspect (bytes-only) so CI can catch a would-be fake match
    sys.exit(1 if counts.get("bytes-only") else 0)

if __name__ == "__main__":
    main()
