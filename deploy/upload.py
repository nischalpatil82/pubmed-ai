#!/usr/bin/env python3
"""
Push this project to Hugging Face: the data to a dataset repo, the app to a
Space. Run it from the project root.

    set HF_TOKEN=hf_xxx
    python deploy/upload.py --user Nischalpatil            # both
    python deploy/upload.py --user Nischalpatil --only data
    python deploy/upload.py --user Nischalpatil --only space

The data upload is ~931 MB and is the slow part; the Space is a few hundred KB.
Re-running is cheap - HF skips files whose hash already matches, so an
interrupted upload resumes rather than restarting.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def size_mb(*paths: str) -> float:
    t = 0
    for p in paths:
        for r, _, fs in os.walk(p):
            for f in fs:
                t += os.path.getsize(os.path.join(r, f))
    return t / 1e6


def push_data(api, user: str, name: str) -> str:
    repo = f"{user}/{name}"
    store, index = os.path.join(ROOT, "store"), os.path.join(ROOT, "index")
    for p in (store, index):
        if not os.path.isdir(p):
            sys.exit(f"missing {p} - build the store before deploying")

    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=False)
    print(f"dataset {repo}: uploading {size_mb(store, index):,.0f} MB "
          f"(this is the slow step)", flush=True)

    # Upload the two folders separately so a failure in one does not lose the
    # other, and so progress is visible between them.
    for label, path in (("store", store), ("index", index)):
        print(f"  -> {label}/", flush=True)
        api.upload_folder(repo_id=repo, repo_type="dataset",
                          folder_path=path, path_in_repo=label,
                          commit_message=f"add {label}")
    print(f"dataset ready: https://huggingface.co/datasets/{repo}")
    return repo


def push_space(api, user: str, name: str, data_repo: str, reuse: bool = False) -> str:
    """
    Push the app into a Space.

    reuse=True targets an EXISTING Space instead of creating one. Hugging Face
    now refuses to create Docker Spaces on the free tier (402), but Spaces made
    before that change keep working and still accept pushes. So an existing
    free Docker Space is the only way onto free CPU without PRO.
    """
    repo = f"{user}/{name}"
    if reuse:
        info = api.space_info(repo)
        print(f"reusing existing space {repo} (sdk={info.sdk})")
        if info.sdk != "docker":
            sys.exit(f"{repo} is a {info.sdk} space; this app needs sdk=docker")
    else:
        api.create_repo(repo_id=repo, repo_type="space", exist_ok=True,
                        space_sdk="docker", private=False)

    # Assemble exactly what the Space needs - not the whole project. raw/,
    # logs/, store/ and index/ must not go here: the first two are irrelevant
    # and the last two live in the dataset repo.
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(os.path.join(ROOT, "pipeline"), os.path.join(tmp, "pipeline"),
                        ignore=shutil.ignore_patterns("__pycache__", "_unused_sql"))
        os.makedirs(os.path.join(tmp, "deploy"), exist_ok=True)
        shutil.copy(os.path.join(ROOT, "deploy", "fetch_data.py"),
                    os.path.join(tmp, "deploy", "fetch_data.py"))
        shutil.copy(os.path.join(ROOT, "deploy", "Dockerfile"), os.path.join(tmp, "Dockerfile"))
        shutil.copy(os.path.join(ROOT, "deploy", "README.md"), os.path.join(tmp, "README.md"))
        shutil.copy(os.path.join(ROOT, "requirements.txt"), os.path.join(tmp, "requirements.txt"))

        # Never ship the key. It belongs in the Space's secrets, not in git.
        for stray in ("llm.env",):
            p = os.path.join(tmp, stray)
            if os.path.exists(p):
                os.remove(p)

        print(f"space {repo}: uploading {size_mb(tmp):,.1f} MB", flush=True)
        # Clear the previous app's files so the two do not interleave. The Space
        # is a git repo, so the old version stays in history.
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=tmp,
                          commit_message="deploy pubmed literature intelligence",
                          delete_patterns=["*"])

    print(f"space ready: https://huggingface.co/spaces/{repo}")
    return repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--space-name", default="pubmed-ai")
    ap.add_argument("--data-name", default="pubmed-ai-data")
    ap.add_argument("--only", choices=["data", "space"])
    ap.add_argument("--reuse", action="store_true",
                    help="push into an existing Space instead of creating one")
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN first - create one at "
                 "https://huggingface.co/settings/tokens with WRITE access")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    who = api.whoami()
    print(f"authenticated as {who.get('name')}\n")

    data_repo = f"{a.user}/{a.data_name}"
    if a.only != "space":
        data_repo = push_data(api, a.user, a.data_name)
    if a.only != "data":
        push_space(api, a.user, a.space_name, data_repo, reuse=a.reuse)

    print("\nNext, in the Space settings -> Variables and secrets, add:")
    print(f"  PUBMED_DATA_REPO   = {data_repo}")
    print( "  PUBMED_LLM         = cloud")
    print( "  PUBMED_API_BASE    = https://api.groq.com/openai/v1")
    print( "  PUBMED_CLOUD_MODEL = openai/gpt-oss-120b")
    print( "  PUBMED_API_KEY     = <your groq key>   (as a SECRET, not a variable)")


if __name__ == "__main__":
    main()
