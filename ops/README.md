# Windows deployment and automatic code updates

Each Windows deployment runs its **own copy** of the application and the verified
PubMed snapshot. Development happens in a separate Git checkout.

```text
Development checkout: edit -> test -> commit -> push covid-files to GitHub
                                              |
                                              v
Deployment PC: check GitHub every 5 minutes -> fast-forward code -> restart app
```

Saving a local file does **not** change another PC. Pushing a tested commit
does. This keeps unfinished edits from interrupting a running deployment.
The existing GitHub workflow separately updates the Hugging Face Space.

## Prerequisites

Use the approved company VPN and Remote Desktop. On the deployment PC, check that the
Windows account can reach GitHub, that Git and Python 3.11 are installed, and
that it has enough disk for the verified 18.66 GB release **plus** Python/model
caches and free working space. Choose where the application and
dataset should be stored. Keep RDP behind the VPN; the app listens only on
`127.0.0.1:8010` and can be opened in a browser on that remote desktop.

The data is intentionally absent from Git. Transfer the verified
`covid-files/dataset.json`, `stores/<snapshot>/`, and `indexes/<snapshot>/`
from the existing release, or download the **same pinned release** from the
Hugging Face dataset repository. Verify the release inventory/checksums before
serving. Do not ingest XML or recreate embeddings on the deployment PC.

## First installation on Windows

Run these in PowerShell from the chosen parent directory. Replace the example
location with the selected deployment location.

```powershell
git clone --branch covid-files https://github.com/nischalpatil82/pubmed-ai.git
Set-Location .\pubmed-ai
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the verified snapshot under `covid-files/`. If a company-approved download
from the existing Hugging Face dataset is easier, `deploy/fetch_data.py` can
download the **pinned** revision and verify its checksums. It needs
`huggingface_hub` plus `PUBMED_DATA_REPO`, the exact 40-character
`PUBMED_DATA_REVISION`, and `PUBMED_DATA_DIR` set on that PC. If the dataset is
private, its read token must be supplied on that PC as `HF_TOKEN`. Point
`--dataset` at the resulting `PUBMED_DATA_DIR\dataset.json`; do not download a
different revision. Check disk headroom before starting an 18.66 GB transfer.
For a copied release, copy `release.json` too and verify every file once:

```powershell
.\.venv\Scripts\python.exe ops\verify_snapshot.py
```

If the data is in a different directory, pass its `dataset.json` and
`release.json` paths with `--dataset` and `--release`. A pinned download through
`deploy/fetch_data.py` already performs this checksum verification.

Put the deployment's approved answer-model settings in a local `llm.env` based on
`llm.env.example`, or leave cloud settings out and run an installed local
Ollama model. `llm.env` is ignored by Git and is never copied by the updater.
Do not send a key in chat or commit it. Groq GPT OSS 120B needs no local
generative model. For hybrid semantic search on a fresh PC, allow the pinned
BGE query model to download once by setting `PUBMED_ALLOW_MODEL_DOWNLOAD=1`
in `llm.env` if the company's network policy permits it; otherwise the app
discloses that dense retrieval is unavailable and continues with keyword search.

Start once by hand and open `http://127.0.0.1:8010` **inside the remote desktop**:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_all.ps1
```

After checking Search, Research overview, Evidence, and Ask, stop that manual
server with Ctrl+C. Then check the automatic updater without starting it:

```powershell
.\.venv\Scripts\python.exe ops\local_sync.py --check
```

It must report `Local dataset ready: True` and either `Up to date` or a
fast-forward update. It refuses a dirty checkout, a different branch, a
diverged branch, and a code update that changes `requirements.txt`.

## Start updates automatically

After the manual checks, create one Windows **Task Scheduler** task under the
deployment PC's authorised Windows account:

| Task field | Value |
|---|---|
| Name | `PubMed AI local sync` |
| Trigger | At logon of the account that owns this checkout |
| Program | Full path to `pubmed-ai\.venv\Scripts\python.exe` |
| Arguments | Full path to `pubmed-ai\ops\local_sync.py`; add `--dataset "C:\path\to\dataset.json"` if data is elsewhere |
| Start in | Full path to the `pubmed-ai` folder |
| If already running | Do not start a new instance |

Use the normal user account, not an administrator or SYSTEM task. That account
must have access to the code, data, `llm.env`, and GitHub. Choose a Task
Scheduler sign-in mode compatible with the company's policies; if it runs only
while the user is signed in, disconnecting RDP is fine but signing out ends it.
Keep the PC powered and connected. The script starts the app, checks GitHub every
five minutes, and restarts **only its own app process** after a safe fast-forward.
Updates normally appear after the next check plus app startup time. Use
`ops/logs/sync.log` and `ops/logs/server.log` to diagnose failures.

Run only **one** copy of the supervisor. Stop any manually started app on port
8010 before starting the task. Never edit tracked code on the deployment PC; push
changes from the development checkout instead. If the deployment PC has local tracked edits, the
updater pauses rather than overwriting them.

## What synchronises and what needs a separate step

- Code, frontend, and documentation on `covid-files` update after a successful
  push. A bad push can break the deployed copy; test first and use a new revert
  commit if rollback is needed. Do not force-push its branch.
- `llm.env`, model downloads, logs, and the 18.66 GB dataset stay local to the
  deployment PC. A new dataset release needs a deliberate transfer, checksum check,
  and manifest switch. Code-only changes do not require new embeddings.
- A push changing `requirements.txt` is held for a supervised dependency
  installation on that PC. This avoids replacing Python packages under a
  running service. Stop the Task Scheduler task, run `git fetch origin
  covid-files`, `git merge --ff-only origin/covid-files`, and
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then start
  the task again after the install succeeds.

The task's installation and running state must be checked on the deployment PC;
the presence of `local_sync.py` in Git does not confirm it is enabled there.

## Enable the current updater on D:

Once the tested changes have been committed and pushed, stop the manually started
server with Ctrl+C. If an older supervisor is running as the scheduled task, stop
that task first. In PowerShell on the deployment PC:

```powershell
Set-Location D:\pubmed-ai
git pull --ff-only origin covid-files
powershell -NoProfile -ExecutionPolicy Bypass -File .\ops\enable_local_sync.ps1
Get-ScheduledTask -TaskName 'PubMed AI local sync' | Select-Object TaskName, State
Get-Content .\ops\logs\sync.log -Tail 15
```

The helper registers a normal-user task at logon and starts it now. It refuses
to replace an unrelated task or compete with an app already using port 8010.
The interactive task requires that account to remain signed in; disconnecting
AnyDesk does not sign out of Windows. Keep the machine on and online.
After the log reports app startup, verify readiness inside that desktop with
`Invoke-RestMethod http://127.0.0.1:8010/health` and open the app in its browser.

The updated supervisor prepares the citation summary and the compact article
lookup before starting its app. Preparation is idempotent and is checked again
after updates: unchanged indexes are reused. First preparation may take several
minutes and needs roughly 2.2 GB of additional disk for this snapshot, plus the
2 GB reserve. It does not recreate embeddings, change the dataset manifest, or
download the dataset. If preparation fails, it logs the problem and serves using
the source-file fallback. Set `PUBMED_PREPARE_PERFORMANCE=0` in `llm.env` to skip
automatic preparation and run `manage.py performance` manually instead.

Future tested commits pushed to `covid-files` are fetched every five minutes.
Dependency changes and dataset replacements still need the separate steps above.
When the supervisor script itself changes, restart the scheduled task once so
the running supervisor loads its updated code.
