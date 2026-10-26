"""FactShield v2 — local server for the browser extension.

    uvicorn fsv2.api:app --port 8000                  # real models and search
    FSV2_STUB=1 uvicorn fsv2.api:app --port 8000      # offline: fake search,
                                                      # fake models, fake verdicts

Endpoints
    GET  /v2/health          what is loaded, and whether this is the offline stub
    POST /v2/extract         highlighted text -> claims ("Checking as: ...")
    POST /v2/check           up to 3 claims -> a job id (checks run in the background)
    GET  /v2/jobs/{job_id}   progress steps and results, polled by the extension

Claims in a job are checked one after another, not in parallel: the Groq free
plan allows 8,000 tokens per minute, about two panel calls.

Every check calls the three panel models (3 calls, up to 9 when they
disagree, plus one for the explanation) and Tavily (2 searches, up to 4).
With the Groq free plan's 200,000 tokens a day, that is roughly 15–40 claims
a day, shared with any thesis runs on the same key.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import fsv2  # noqa: F401  (sets sys.path)

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import checker as chk
from . import live
from .settings import cfg

STUB = os.environ.get("FSV2_STUB", "") not in ("", "0", "false", "False")
STATE: dict = {"jobs": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    STATE["checker"] = chk.load(stub=STUB)
    STATE["checker_seconds"] = round(time.perf_counter() - t0, 1)
    STATE["client"] = httpx.AsyncClient()
    STATE["search"] = live.StubSearch() if STUB else live.Search()
    yield
    await STATE["client"].aclose()


app = FastAPI(title="FactShield v2", lifespan=lifespan)
# The extension calls through its background worker (no CORS needed); this is
# for local test pages only. The server listens on localhost.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Context(BaseModel):
    selection: str = ""
    surrounding_text: str = ""
    page_title: str | None = None
    page_published: str | None = None
    page_domain: str | None = None


class ClaimIn(BaseModel):
    checking_as: str = Field(..., min_length=3, max_length=600)
    queries: list[str] = []


class CheckIn(BaseModel):
    claims: list[ClaimIn]
    context: Context | None = None


@app.get("/v2/health")
async def health():
    from study.common import config as sc
    m = sc()["models"]
    return {"ok": True, "stub": STUB, "version": cfg()["version"], "checker": STATE["checker"].name,
            "checker_load_seconds": STATE.get("checker_seconds"),
            "panel": {k: m[k]["model"] for k in cfg()["panel"]},
            "limits": {k: cfg()["live"][k] for k in ("word_cap", "max_claims_per_check")}}


@app.post("/v2/extract")
async def extract(ctx: Context):
    try:
        out = await live.extract(ctx.model_dump(), stub=STUB, client=STATE["client"])
    except live.TooLong as exc:
        raise HTTPException(413, {"error": "too_long", "words": exc.n, "max": exc.cap})
    out["stub"] = STUB
    return out


@app.post("/v2/check")
async def check(body: CheckIn):
    cap = cfg()["live"]["max_claims_per_check"]
    if not body.claims:
        raise HTTPException(400, {"error": "no_claims"})
    if len(body.claims) > cap:
        raise HTTPException(400, {"error": "too_many_claims", "max": cap})
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "stub": STUB, "started": time.time(),
           "claims": [{"checking_as": c.checking_as, "steps": [], "result": None, "error": None,
                       "status": "waiting"} for c in body.claims]}
    STATE["jobs"][job_id] = job
    asyncio.create_task(_run(job, body))
    return {"job_id": job_id}


async def _run(job: dict, body: CheckIn) -> None:
    for slot, c in zip(job["claims"], body.claims):
        slot["status"] = "running"

        def progress(msg: str, slot=slot):
            slot["steps"].append({"t": round(time.time() - job["started"], 1), "text": msg})

        try:
            res = await live.check_claim(c.checking_as, c.queries, checker=STATE["checker"],
                                         search=STATE["search"], stub=STUB,
                                         client=STATE["client"], progress=progress)
        except Exception as exc:  # shown to the user, never swallowed
            res = {"error": f"{type(exc).__name__}: {exc}"}
        if res.get("error"):
            print(f"[fsv2] check failed for {c.checking_as[:60]!r}: {res['error'][:300]}", flush=True)
            slot["error"] = _friendly(res["error"])
            slot["error_detail"] = res["error"][:600]
            slot["status"] = "error"
        else:
            slot["result"] = res
            slot["status"] = "done"
    job["status"] = "done"


def _friendly(err: str) -> str:
    e = err.lower()
    if "503" in e or "high demand" in e:
        return "One of the checking models is overloaded right now. Please try again in a few minutes."
    if "tokens per day" in e or "(tpd)" in e:
        return "The daily allowance for one of the checking models is used up. Try again later."
    if "429" in e:
        return "Too many requests to one of the checking models. Please wait a minute and try again."
    if "change_me" in e:
        return "A model name is not set in study/config.yaml."
    if "not set" in e:
        return "An API key is missing in .env."
    return "This check failed. Details are in the server log."


@app.get("/v2/jobs/{job_id}")
async def job(job_id: str):
    j = STATE["jobs"].get(job_id)
    if not j:
        raise HTTPException(404, {"error": "unknown_job"})
    return j
