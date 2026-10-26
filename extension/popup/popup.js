/**
 * FactShield 2 — the in-page panel.
 *
 * Flow (design stages 0, 1 and 7):
 *   1. Selection over 800 words  -> say so; nothing is cut silently.
 *   2. Extract claims             -> checklist with editable "Checking as: …";
 *                                    unclear claims greyed with the reason;
 *                                    opinions etc. listed as not checked.
 *                                    A single clear claim skips the list.
 *   3. Check (max 3)              -> live list of steps per claim.
 *   4. Result per claim           -> label + plain reason, 1–3 key quotes with
 *                                    their source, the checked explanation with
 *                                    expandable citations, related fact-checks,
 *                                    all sources, and an honest footer.
 * No confidence percentage is shown anywhere: FactShield has no calibrated
 * score, and a number would suggest a precision it does not have.
 */
(() => {
  const HOST_ID = "factshield-host";
  const WORD_CAP = 800;
  const MAX_CLAIMS = 3;
  const POLL_MS = 1500;

  let host = null, root = null, panel = null, pollTimer = null, stubMode = false;

  const CSS = `
    :host { all: initial; }
    .panel { position:absolute; z-index:2147483647; width:min(440px, calc(100vw - 32px));
      max-height:75vh; overflow:auto; box-sizing:border-box; padding:14px 16px 12px;
      background:#ffffff; color:#1c1c1e; border:1px solid #d4d4d8; border-radius:12px;
      box-shadow:0 10px 32px rgba(0,0,0,.18); font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
    @media (prefers-color-scheme: dark) {
      .panel { background:#1f1f22; color:#ececf0; border-color:#3a3a40; }
      .muted, .from, .meta, .foot, .step { color:#a1a1aa !important; }
      .card { border-color:#3a3a40 !important; background:#26262a !important; }
      textarea { background:#18181b !important; color:#ececf0 !important; border-color:#3a3a40 !important; }
      .quote { background:#2b2b31 !important; }
      .chip { background:#33333a !important; color:#ececf0 !important; }
    }
    * { box-sizing:border-box; }
    .top { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
    .brand { font-weight:700; letter-spacing:.2px; }
    .close { all:unset; cursor:pointer; font-size:20px; line-height:1; padding:2px 6px; border-radius:6px; }
    .close:hover { background:rgba(127,127,127,.18); }
    .stub { background:#fff4cc; color:#5c4400; border:1px solid #e8c95a; border-radius:8px;
      padding:6px 8px; font-size:12px; margin-bottom:10px; }
    .muted { color:#52525b; }
    .err { color:#b42318; }
    .card { border:1px solid #e4e4e7; border-radius:10px; padding:10px; margin:8px 0; background:#fafafa; }
    .card.unclear { opacity:.62; }
    .row { display:flex; gap:8px; align-items:flex-start; }
    textarea { width:100%; min-height:44px; resize:vertical; font:inherit; padding:6px 8px;
      border:1px solid #d4d4d8; border-radius:8px; background:#fff; color:inherit; }
    .from { font-size:12px; color:#52525b; margin-top:4px; }
    .why { font-size:12px; color:#8a5a00; margin-top:4px; }
    button.primary { all:unset; cursor:pointer; background:#1d4ed8; color:#fff; border-radius:8px;
      padding:8px 14px; font-weight:600; margin-top:6px; }
    button.primary[disabled] { background:#a1a1aa; cursor:not-allowed; }
    .skel { height:10px; border-radius:6px; background:linear-gradient(90deg,#e4e4e7,#f4f4f5,#e4e4e7);
      margin:8px 0; animation:sh 1.2s infinite linear; background-size:200% 100%; }
    @keyframes sh { from { background-position:200% 0 } to { background-position:0 0 } }
    .steps { margin:6px 0 0; padding:0; list-style:none; }
    .step { font-size:13px; color:#52525b; padding-left:20px; position:relative; margin:2px 0; }
    .step::before { content:"✓"; position:absolute; left:2px; color:#15803d; }
    .step.now::before { content:"●"; color:#1d4ed8; animation:bl 1s infinite; }
    @keyframes bl { 50% { opacity:.3 } }
    .label { display:inline-flex; gap:6px; align-items:center; font-weight:700; border-radius:8px;
      padding:3px 9px; font-size:14px; }
    .l-supported { background:#dcfce7; color:#14532d; }
    .l-refuted { background:#fee2e2; color:#7f1d1d; }
    .l-not_enough_evidence { background:#e4e4e7; color:#27272a; }
    .l-conflicting_evidence { background:#fef3c7; color:#713f12; }
    .reason { margin:6px 0 4px; }
    .checking { font-size:12px; margin-bottom:6px; }
    h4 { margin:12px 0 4px; font-size:12px; text-transform:uppercase; letter-spacing:.5px; color:#71717a; }
    .quote { margin:6px 0; padding:8px 10px; border-left:3px solid #1d4ed8; background:#f4f4f5; border-radius:0 8px 8px 0; }
    .meta { font-size:12px; color:#52525b; margin-top:3px; }
    a { color:#1d4ed8; }
    .sent { margin:4px 0; }
    .sent.summary { font-style:italic; }
    .chip { all:unset; cursor:pointer; font-size:11px; background:#e4e4e7; color:#27272a;
      border-radius:5px; padding:0 5px; margin-left:3px; }
    .cited { font-size:12px; margin:4px 0 6px 10px; padding:6px 8px; border-left:2px solid #a1a1aa; }
    details { margin-top:6px; } summary { cursor:pointer; font-size:13px; }
    .src { font-size:12px; margin:5px 0; }
    .foot { font-size:12px; color:#52525b; border-top:1px solid rgba(127,127,127,.25); margin-top:10px; padding-top:8px; }
    .warn { font-size:12px; color:#8a5a00; margin-top:6px; }
  `;

  const LABEL_GLYPH = { supported: "✓", refuted: "✗", not_enough_evidence: "?", conflicting_evidence: "⇄" };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ------------------------------------------------------------- plumbing

  function api(method, path, body) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage({ type: "FACTSHIELD_API", method, path, body }, (r) => {
          if (chrome.runtime.lastError || !r) resolve({ ok: false, status: 0, error: "unreachable" });
          else resolve(r);
        });
      } catch {
        resolve({ ok: false, status: 0, error: "unreachable" });
      }
    });
  }

  function mount(anchor) {
    close();
    host = document.createElement("div");
    host.id = HOST_ID;
    host.style.cssText = "all:initial;position:absolute;top:0;left:0;width:0;height:0;";
    document.documentElement.appendChild(host);
    root = host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = CSS;
    root.appendChild(style);
    panel = document.createElement("div");
    panel.className = "panel";
    const left = Math.max(8, Math.min(anchor.left, window.scrollX + window.innerWidth - 456));
    panel.style.top = `${anchor.top + 8}px`;
    panel.style.left = `${left}px`;
    root.appendChild(panel);
  }

  function close() {
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = null;
    host?.remove();
    host = root = panel = null;
  }

  function frame(inner) {
    if (!panel) return;
    panel.innerHTML = `
      <div class="top"><div class="brand">FactShield</div><button class="close" title="Close (Esc)">×</button></div>
      ${stubMode ? `<div class="stub"><b>Offline test mode:</b> fake search and fake models. These verdicts mean nothing.</div>` : ""}
      ${inner}`;
    panel.querySelector(".close").onclick = close;
  }

  function problem(text) {
    frame(`<p class="err">${esc(text)}</p>`);
  }

  const UNREACHABLE = "Cannot reach the FactShield server. Is it running? (uvicorn fsv2.api:app --port 8000)";

  // --------------------------------------------------------- 1. extract

  async function start() {
    const payload = window.__factshieldCapture?.();
    if (!payload) return;
    mount(window.__factshieldAnchor());
    stubMode = false;

    if (payload.words > WORD_CAP) {
      problem(`Your selection is ${payload.words} words. FactShield checks up to ${WORD_CAP} words at a time. Please select a shorter part.`);
      return;
    }
    frame(`<p class="muted">Finding the claims in your selection…</p><div class="skel"></div><div class="skel" style="width:70%"></div>`);

    const r = await api("POST", "/v2/extract", payload);
    if (!panel) return;
    if (!r.ok) {
      if (r.status === 413) problem(`Your selection is ${r.data?.detail?.words} words. FactShield checks up to ${r.data?.detail?.max} words at a time.`);
      else problem(r.status === 0 ? UNREACHABLE : `The server returned an error (${r.status}).`);
      return;
    }
    const data = r.data;
    stubMode = !!data.stub;
    if (data.error) { problem(`Couldn't read the claims in this selection: ${data.error}`); return; }

    if (!data.claims.length) {
      frame(`<p>I didn't find a checkable factual claim in this selection.</p>${skippedHtml(data.skipped)}`);
      return;
    }
    if (data.claims.length === 1 && data.claims[0].resolved && !data.skipped.length) {
      runCheck([data.claims[0]], payload);
      return;
    }
    checklist(data, payload);
  }

  function skippedHtml(skipped) {
    if (!skipped?.length) return "";
    return `<h4>Not checked</h4>` + skipped.map((s) =>
      `<div class="card unclear"><div>${esc(s.text)}</div><div class="why">${esc(s.reason)}</div></div>`).join("");
  }

  // ------------------------------------------------------ 2. checklist

  function checklist(data, payload) {
    const cards = data.claims.map((c, i) => `
      <div class="card ${c.resolved ? "" : "unclear"}" data-i="${i}">
        <div class="row">
          <input type="checkbox" ${c.resolved && i < MAX_CLAIMS ? "checked" : ""} aria-label="Check this claim">
          <div style="flex:1">
            <div class="checking muted">Checking as:</div>
            <textarea>${esc(c.checking_as)}</textarea>
            ${c.original && c.original !== c.checking_as ? `<div class="from">From the page: “${esc(c.original)}”</div>` : ""}
            ${c.resolved ? "" : `<div class="why">Unclear: ${esc(c.unclear || "a reference could not be resolved")}. Edit the text to make it clear, then tick it.</div>`}
          </div>
        </div>
      </div>`).join("");
    frame(`
      <p class="muted">Choose up to ${MAX_CLAIMS} claims to check. You can edit how each one is checked.</p>
      ${cards}
      ${skippedHtml(data.skipped)}
      <button class="primary go"></button>`);

    const boxes = [...panel.querySelectorAll('input[type="checkbox"]')];
    const go = panel.querySelector(".go");
    const refresh = () => {
      const n = boxes.filter((b) => b.checked).length;
      boxes.forEach((b) => { b.disabled = !b.checked && n >= MAX_CLAIMS; });
      go.textContent = n ? `Check ${n} selected` : "Select a claim to check";
      go.disabled = !n;
    };
    boxes.forEach((b) => (b.onchange = refresh));
    panel.querySelectorAll(".card textarea").forEach((ta) => {
      ta.oninput = () => ta.closest(".card").classList.remove("unclear");
    });
    refresh();
    go.onclick = () => {
      const chosen = [...panel.querySelectorAll(".card[data-i]")]
        .filter((card) => card.querySelector('input[type="checkbox"]').checked)
        .map((card) => {
          const c = data.claims[+card.dataset.i];
          const text = card.querySelector("textarea").value.trim();
          return { ...c, checking_as: text, queries: text === c.checking_as ? c.queries : [text] };
        })
        .filter((c) => c.checking_as.length >= 3);
      if (chosen.length) runCheck(chosen, payload);
    };
  }

  // ---------------------------------------------------- 3. check + poll

  async function runCheck(claims, payload) {
    frame(`<p class="muted">Starting…</p>`);
    const r = await api("POST", "/v2/check", {
      claims: claims.map((c) => ({ checking_as: c.checking_as, queries: c.queries || [] })),
      context: payload,
    });
    if (!panel) return;
    if (!r.ok) { problem(r.status === 0 ? UNREACHABLE : `Couldn't start the check (${r.status}).`); return; }
    poll(r.data.job_id);
  }

  async function poll(jobId) {
    const r = await api("GET", `/v2/jobs/${jobId}`);
    if (!panel) return;
    if (!r.ok) { problem(r.status === 0 ? UNREACHABLE : `Lost track of the check (${r.status}).`); return; }
    const job = r.data;
    stubMode = !!job.stub;
    renderJob(job);
    if (job.status !== "done") pollTimer = setTimeout(() => poll(jobId), POLL_MS);
  }

  function claimHtml(c, i) {
    if (c.status === "done") return resultHtml(c.result, i);
    if (c.status === "error") return `<div class="card"><div class="checking muted">Checking as: ${esc(c.checking_as)}</div><p class="err">${esc(c.error)}</p></div>`;
    const steps = c.steps.map((s, j) =>
      `<li class="step ${j === c.steps.length - 1 ? "now" : ""}">${esc(s.text)}</li>`).join("");
    return `<div class="card"><div class="checking muted">Checking as: ${esc(c.checking_as)}</div>
      ${c.status === "waiting" ? `<div class="muted">Waiting for the claim before…</div>` : `<ul class="steps">${steps || '<li class="step now">Starting</li>'}</ul>`}</div>`;
  }

  /** Each claim has its own slot, re-drawn only when its content changes, so
   *  an opened citation or source list stays open while other claims run. */
  function renderJob(job) {
    if (!panel.querySelector(".jobwrap")) {
      frame(`<div class="jobwrap">${job.claims.map((_, i) => `<div data-c="${i}"></div>`).join("")}</div>`);
    }
    job.claims.forEach((c, i) => {
      const el = panel.querySelector(`[data-c="${i}"]`);
      const html = claimHtml(c, i);
      if (el.__html === html) return;
      el.__html = html;
      el.innerHTML = html;
      if (c.status === "done") wireCitations(el, c.result, i);
    });
  }

  // --------------------------------------------------------- 4. result

  function ratingText(s) {
    if (s.source_type === "fact-check") return "fact-check";
    const t = s.rating == null ? "unrated" : `news-quality ${Number(s.rating).toFixed(2)}/1`;
    return `${s.source_type} · ${t}`;
  }

  function resultHtml(r, i) {
    const src = Object.fromEntries(r.sources.map((s) => [s.pid, s]));
    const quotes = r.key_quotes.map((q) => {
      const s = src[q.pid] || {};
      return `<div class="quote">“${esc(q.quote)}”
        <div class="meta">${esc(s.domain)} · ${esc(ratingText(s))}${s.date ? ` · ${esc(s.date)}` : ""} ·
          <a href="${esc(s.url)}" target="_blank" rel="noopener">Open source →</a></div></div>`;
    }).join("");

    let expl = "";
    if (r.quotes_only && r.label !== "not_enough_evidence") {
      expl = `<div class="warn">The written explanation didn't pass the citation check, so only the quotes are shown.</div>`;
    } else if (r.explanation.length) {
      expl = r.explanation.map((x, j) => {
        if (x.kind === "summary") {
          return `<div class="sent summary"><span class="muted">FactShield's summary:</span> ${esc(x.text.replace(/^summary:\s*/i, ""))}</div>`;
        }
        const text = x.text.replace(/\s*[\[【［(（]\s*P\d+(?:\s*[,;]\s*P\d+)*\s*[\]】］)）]/g, "");
        const chips = x.passages.map((p) =>
          `<button class="chip" data-claim="${i}" data-sent="${j}" data-pid="${esc(p)}" title="Show the passage">${esc(src[p]?.domain || p)}</button>`).join("");
        return `<div class="sent">${esc(text)}${chips}<div class="cited" hidden data-slot="${i}-${j}"></div></div>`;
      }).join("");
      expl = `<h4>Why</h4>${expl}`;
    }

    const related = r.related_factchecks.length ? `<h4>Fact-checks of related claims (not used as evidence)</h4>` +
      r.related_factchecks.slice(0, 3).map((f) => `<div class="src">${esc(f.publisher)} rated “${esc(f.reviewed_claim)}”: <b>${esc(f.rating)}</b> · <a href="${esc(f.url)}" target="_blank" rel="noopener">open</a></div>`).join("") : "";

    const sources = r.sources.length ? `<details data-k="src-${i}"><summary>All ${r.sources.length} sources read</summary>` +
      r.sources.map((s) => `<div class="src">${esc(s.domain)} · ${esc(ratingText(s))}${s.date ? ` · ${esc(s.date)}` : ""}${s.copy_of ? ` · same text as ${esc(src[s.copy_of]?.domain || s.copy_of)}` : ""} · <a href="${esc(s.url)}" target="_blank" rel="noopener">open</a></div>`).join("") + `</details>` : "";

    const warn = r.search_errors?.length ? `<div class="warn">Some searches failed: ${esc(r.search_errors.join("; "))}</div>` : "";

    return `<div class="card">
      <div class="checking muted">Checking as: ${esc(r.checking_as)}</div>
      <span class="label l-${esc(r.label)}">${LABEL_GLYPH[r.label] || ""} ${esc(r.label_text)}</span>
      <div class="reason">${esc(r.reason_text)}</div>
      ${quotes ? `<h4>Key evidence</h4>${quotes}` : ""}
      ${expl}${related}${sources}${warn}
      <div class="foot">Checked against ${r.n_sources} source${r.n_sources === 1 ? "" : "s"} found on ${esc(r.checked_on)}. FactShield can be wrong — read the quotes.</div>
    </div>`;
  }

  function wireCitations(el, r, i) {
    el.querySelectorAll(".chip").forEach((b) => {
      b.onclick = () => {
        const s = r.sources.find((x) => x.pid === b.dataset.pid);
        const slot = el.querySelector(`[data-slot="${i}-${b.dataset.sent}"]`);
        if (!s || !slot) return;
        const same = !slot.hidden && slot.dataset.pid === s.pid;
        slot.hidden = same;
        slot.dataset.pid = s.pid;
        slot.innerHTML = `${esc(s.text)}<div class="meta">${esc(s.domain)} · <a href="${esc(s.url)}" target="_blank" rel="noopener">open</a></div>`;
      };
    });
  }

  // ------------------------------------------------------------- entry

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type === "FACTSHIELD_START") start();
    return false;
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
})();
