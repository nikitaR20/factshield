/**
 * Result panel.
 *
 * Rendered into a CLOSED shadow root. Host pages routinely define aggressive
 * global CSS; without isolation the panel breaks on a meaningful fraction of
 * sites, and our styles would leak into theirs.
 *
 * TWO LAYOUTS, ONE BACKEND. The `ui_condition` assigned at install decides
 * which renders. Identical data, identical verdicts — only the order and
 * prominence differ:
 *
 *   evidence_first : sources are primary, verdict sits quietly below
 *   verdict_first  : verdict is a large banner, sources collapsed behind a click
 *
 * Holding the backend identical is what makes any behavioural difference
 * attributable to presentation alone.
 */

(() => {
  const HOST_ID = "factshield-host";
  let host, root, openedAt, currentRequestId, sourcesOpened = false;

  // ------------------------------------------------------------------ styles

  const CSS = `
    :host { all: initial; }
    .panel {
      position: absolute; z-index: 2147483647; width: 380px; max-width: 92vw;
      max-height: 70vh; overflow-y: auto;
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #fff; color: #1a1a1a; border: 1px solid #d8d8d8;
      border-radius: 10px; box-shadow: 0 8px 28px rgba(0,0,0,.16); padding: 14px 16px;
    }
    @media (prefers-color-scheme: dark) {
      .panel { background: #1c1c1e; color: #ececec; border-color: #3a3a3c; }
      .claim, .meta { background: #2c2c2e; }
      .src { border-color: #3a3a3c; }
    }
    .top { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
    .brand { font-size:11px; letter-spacing:.08em; text-transform:uppercase; opacity:.55; }
    .close { cursor:pointer; border:0; background:none; font-size:18px; line-height:1; opacity:.5; color:inherit; }
    .close:hover { opacity:1; }

    /* The resolved claim is shown FIRST and always. If pronoun resolution went
       wrong, the user catches it here rather than receiving a confident verdict
       about a claim they never made. */
    .claim { background:#f4f4f5; border-radius:6px; padding:8px 10px; margin-bottom:10px; font-size:13px; }
    .claim b { display:block; font-size:10px; text-transform:uppercase; letter-spacing:.06em; opacity:.55; margin-bottom:3px; }
    .warn { font-size:12px; color:#8a5a00; margin-bottom:8px; }

    .meta { font-size:12px; opacity:.75; background:#f4f4f5; border-radius:6px; padding:6px 9px; margin-bottom:10px; }

    .src { border:1px solid #e6e6e6; border-radius:7px; padding:9px 10px; margin-bottom:7px; }
    .src-top { display:flex; justify-content:space-between; gap:8px; font-size:11px; opacity:.7; margin-bottom:4px; }
    .quote { font-size:13px; margin:0 0 6px; }
    .src a { font-size:12px; color:#0b5fd0; text-decoration:none; }
    .src a:hover { text-decoration:underline; }

    /* Colour is reinforcement only, never the sole carrier: every verdict also
       has a text label and a distinct glyph. Red/green alone fails WCAG 1.4.1
       Level A, and red-green deficiency is the most common form. */
    .tag { display:inline-flex; align-items:center; gap:5px; font-weight:600;
           border-radius:5px; padding:2px 8px; font-size:12px; }
    .t-supported { background:#e3f4e6; color:#14532d; }
    .t-refuted   { background:#fde8e8; color:#7f1d1d; }
    .t-partly    { background:#fef3d6; color:#78350f; }
    .t-unresolved{ background:#ececef; color:#3f3f46; }

    .axes { display:flex; flex-direction:column; gap:6px; margin:10px 0; }
    .axis { display:flex; justify-content:space-between; align-items:center; gap:10px; }
    .axis span:first-child { font-size:12px; opacity:.7; }

    .banner { text-align:center; border-radius:8px; padding:14px 10px; margin-bottom:10px; }
    .banner .big { font-size:22px; font-weight:700; letter-spacing:-.01em; }
    .banner .sub { font-size:12px; opacity:.8; margin-top:3px; }

    .expl { font-size:13px; margin:8px 0 0; }
    .toggle { cursor:pointer; background:none; border:0; color:#0b5fd0; font-size:12px;
              padding:6px 0; font-family:inherit; }
    .score { font-variant-numeric: tabular-nums; font-size:12px; opacity:.7; }
    .skeleton { height:12px; border-radius:4px; background:linear-gradient(90deg,#eee,#f6f6f6,#eee);
                background-size:200% 100%; animation:sh 1.2s infinite; margin:7px 0; }
    @keyframes sh { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
    .err { font-size:13px; color:#7f1d1d; }
  `;

  const VERDICT_LABEL = {
    supported: ["Supported", "✓", "t-supported"],
    refuted: ["Refuted", "✕", "t-refuted"],
    partly_supported: ["Partly supported", "◐", "t-partly"],
    unresolved: ["Cannot determine", "?", "t-unresolved"],
  };

  const STANDING_LABEL = {
    consistent: "Evidence matches the claim",
    overstated: "Weaker evidence than the claim implies",
    outdated: "Was supported; evidence has moved",
    contested: "Credible sources disagree",
    unestablished: "No evidence of the expected kind yet",
    amplified: "Widely repeated, one original source",
  };

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ------------------------------------------------------------------ mount

  function mount(anchor) {
    document.getElementById(HOST_ID)?.remove();
    host = document.createElement("div");
    host.id = HOST_ID;
    host.style.cssText = "all:initial;position:absolute;top:0;left:0;width:0;height:0;";
    document.body.appendChild(host);

    root = host.attachShadow({ mode: "closed" });
    const style = document.createElement("style");
    style.textContent = CSS;
    root.appendChild(style);

    const panel = document.createElement("div");
    panel.className = "panel";
    panel.style.top = `${anchor.top + 8}px`;
    panel.style.left = `${Math.min(anchor.left, window.innerWidth - 400)}px`;
    root.appendChild(panel);

    openedAt = Date.now();
    sourcesOpened = false;
    return panel;
  }

  function close() {
    if (openedAt && currentRequestId) {
      emit({
        event: "panel_closed",
        request_id: currentRequestId,
        ms_open: Date.now() - openedAt,
        sources_opened: sourcesOpened,
      });
    }
    host?.remove();
    host = root = null;
    openedAt = currentRequestId = null;
  }

  function emit(event) {
    chrome.runtime.sendMessage({ type: "FACTSHIELD_EVENT", event });
  }

  function chrome_() { return chrome; }

  // ----------------------------------------------------------------- render

  function shell(panel, claimText) {
    panel.innerHTML = `
      <div class="top">
        <div class="brand">FactShield</div>
        <button class="close" title="Close">×</button>
      </div>
      <div class="claim"><b>Checking</b>${esc(claimText)}</div>
      <div class="skeleton" style="width:85%"></div>
      <div class="skeleton" style="width:65%"></div>
      <div class="skeleton" style="width:75%"></div>`;
    panel.querySelector(".close").onclick = close;
  }

  function sourceCard(e) {
    const stance =
      e.stance === "supports" ? "agrees" :
      e.stance === "refutes" ? "disagrees" : "unclear";
    const role = e.role === "assesses" ? "checked this" : "repeats the claim";
    return `
      <div class="src">
        <div class="src-top">
          <span>[${e.id}] ${esc(e.source_domain)} · ${esc(e.tier.replace(/_/g, " "))}</span>
          <span>${esc(e.published_date || "no date")}</span>
        </div>
        <p class="quote">${esc(e.quote)}</p>
        <div class="src-top"><span>${stance} · ${role}</span>
          <a href="${esc(e.url)}" target="_blank" rel="noopener" data-src="${e.id}">Open source →</a>
        </div>
      </div>`;
  }

  function axesBlock(v) {
    const [label, glyph, cls] = VERDICT_LABEL[v.claim_verdict] || VERDICT_LABEL.unresolved;
    const score = v.support_score === null || v.support_score === undefined
      ? "" : `<span class="score">${v.support_score}/100</span>`;
    return `
      <div class="axes">
        <div class="axis"><span>Are the facts supported?</span>
          <span class="tag ${cls}">${glyph} ${label}</span></div>
        <div class="axis"><span>How good is the evidence?</span>
          <span class="score">${esc(STANDING_LABEL[v.evidence_standing] || v.evidence_standing)}</span></div>
        <div class="axis"><span>Evidence-weighted support</span>${score || '<span class="score">not enough evidence to score</span>'}</div>
      </div>`;
  }

  function render(panel, data, condition) {
    currentRequestId = data.request_id;

    if (!data.checkable) {
      panel.innerHTML = `
        <div class="top"><div class="brand">FactShield</div><button class="close">×</button></div>
        <p class="expl">${esc(data.not_checkable_reason)}</p>`;
      panel.querySelector(".close").onclick = close;
      return;
    }

    const claims = data.resolved_claims.map((c) => c.text).join(" · ");
    const warn = data.resolution_confident
      ? ""
      : `<div class="warn">⚠ Some references were ambiguous — check this is what you meant.</div>`;

    const tierCounts = {};
    for (const e of data.evidence) tierCounts[e.tier] = (tierCounts[e.tier] || 0) + 1;
    const meta = `${data.evidence.length} sources · ` +
      Object.entries(tierCounts).map(([t, n]) => `${n} ${t.replace(/_/g, " ")}`).join(" · ");

    const bodies = data.verdicts.map((v) => {
      const evidence = data.evidence.filter((e) => e.sub_claim_id === v.sub_claim_id);
      const srcHtml = evidence.map(sourceCard).join("");
      const expl = v.explanation ? `<p class="expl">${esc(v.explanation)}</p>` : "";
      const multi = data.verdicts.length > 1
        ? `<div class="claim"><b>Claim ${v.sub_claim_id}</b>${esc(v.sub_claim_text)}</div>` : "";

      if (condition === "verdict_first") {
        // Big label, sources hidden behind a click. The low-transparency arm.
        const [label, glyph, cls] = VERDICT_LABEL[v.claim_verdict] || VERDICT_LABEL.unresolved;
        return `${multi}
          <div class="banner ${cls}">
            <div class="big">${glyph} ${label}</div>
            <div class="sub">${esc(STANDING_LABEL[v.evidence_standing] || "")}</div>
          </div>
          ${expl}
          <button class="toggle" data-toggle="${v.sub_claim_id}">Show ${evidence.length} sources ▾</button>
          <div class="sources" data-sources="${v.sub_claim_id}" hidden>${srcHtml}</div>`;
      }

      // evidence_first: sources are primary, verdict sits below them.
      return `${multi}${srcHtml}${axesBlock(v)}${expl}`;
    }).join("<hr style='border:0;border-top:1px solid #eee;margin:12px 0'>");

    panel.innerHTML = `
      <div class="top"><div class="brand">FactShield</div><button class="close">×</button></div>
      <div class="claim"><b>Checking</b>${esc(claims)}</div>
      ${warn}
      <div class="meta">${esc(meta)}</div>
      ${bodies}`;

    panel.querySelector(".close").onclick = close;

    panel.querySelectorAll("[data-toggle]").forEach((btn) => {
      btn.onclick = () => {
        const box = panel.querySelector(`[data-sources="${btn.dataset.toggle}"]`);
        const nowOpen = box.hidden;
        box.hidden = !nowOpen;
        btn.textContent = nowOpen ? "Hide sources ▴" : `Show sources ▾`;
        if (nowOpen && !sourcesOpened) {
          sourcesOpened = true;
          emit({ event: "sources_expanded", request_id: currentRequestId });
        }
      };
    });

    panel.querySelectorAll("[data-src]").forEach((a) => {
      a.onclick = () => emit({
        event: "source_clicked",
        request_id: currentRequestId,
        evidence_id: Number(a.dataset.src),
      });
    });

    if (condition === "evidence_first") sourcesOpened = true; // visible by default
  }

  const ERRORS = {
    rate_limited: "Too many checks in a short time. Wait a moment.",
    timeout: "That took too long. The service may be busy.",
    unreachable: "Cannot reach FactShield. Is the backend running?",
  };

  // ------------------------------------------------------------------ entry

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type !== "FACTSHIELD_START") return false;

    const payload = window.__factshieldCapture?.();
    if (!payload) return false;

    const panel = mount(window.__factshieldAnchor());
    shell(panel, payload.selection);
    emit({ event: "check_requested", domain: payload.page_domain });

    chrome.runtime.sendMessage({ type: "FACTSHIELD_CHECK", payload }, (res) => {
      if (!root) return; // user closed it while waiting
      if (!res?.ok) {
        panel.innerHTML = `
          <div class="top"><div class="brand">FactShield</div><button class="close">×</button></div>
          <p class="err">${esc(ERRORS[res?.error] || "Something went wrong.")}</p>`;
        panel.querySelector(".close").onclick = close;
        return;
      }
      render(panel, res.data, res.uiCondition);
    });
    return false;
  });

  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
})();
