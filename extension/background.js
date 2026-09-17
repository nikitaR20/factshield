/**
 * Background service worker.
 *
 * MV3 terminates this worker after roughly 30 seconds idle and restarts it on
 * the next event. Two consequences shape this file:
 *
 *   1. The context menu is registered in `onInstalled`, NOT at top level.
 *      Re-registering an existing menu id throws, and top-level code runs on
 *      every worker restart.
 *   2. Nothing is kept in memory between events. Participant identity lives in
 *      chrome.storage, which survives restarts.
 */

const MENU_ID = "factshield-check";
const API_BASE = "http://localhost:8000";
const EXTENSION_VERSION = chrome.runtime.getManifest().version;

// ---------------------------------------------------------------- identity

/**
 * Assign a participant id and UI condition once, at install, and never change
 * them. The condition decides which popup layout renders for this person for
 * the life of the install — a between-subjects assignment, because you cannot
 * unsee a layout.
 */
async function ensureIdentity() {
  const stored = await chrome.storage.local.get(["participantId", "uiCondition"]);
  if (stored.participantId && stored.uiCondition) return stored;

  const participantId =
    stored.participantId ||
    `p_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`;

  // Deterministic from the id, so re-running assignment cannot reshuffle it.
  const bucket = [...participantId].reduce((a, ch) => a + ch.charCodeAt(0), 0) % 2;
  const uiCondition = stored.uiCondition || (bucket === 0 ? "evidence_first" : "verdict_first");

  await chrome.storage.local.set({ participantId, uiCondition });
  return { participantId, uiCondition };
}

chrome.runtime.onInstalled.addListener(async () => {
  await ensureIdentity();
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_ID,
      title: "Check with FactShield",
      contexts: ["selection"],
    });
  });
});

// ------------------------------------------------------------------ trigger

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID || !tab?.id) return;
  chrome.tabs.sendMessage(tab.id, { type: "FACTSHIELD_START" });
});

// Keyboard path. Power users during a week-long study will want this, and it
// costs one manifest entry.
chrome.commands?.onCommand.addListener(async (command) => {
  if (command !== "check-selection") return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) chrome.tabs.sendMessage(tab.id, { type: "FACTSHIELD_START" });
});

// ------------------------------------------------------------------ request

/**
 * The content script cannot call the backend directly without tripping the
 * host page's CSP, so the worker proxies it.
 */
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type !== "FACTSHIELD_CHECK") return false;

  (async () => {
    const { participantId, uiCondition } = await ensureIdentity();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);

    try {
      const res = await fetch(`${API_BASE}/v1/check`, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          "X-Participant-Id": participantId,
          "X-Ui-Condition": uiCondition,
          "X-Extension-Version": EXTENSION_VERSION,
        },
        body: JSON.stringify(msg.payload),
      });

      if (res.status === 429) {
        sendResponse({ ok: false, error: "rate_limited" });
        return;
      }
      if (!res.ok) {
        sendResponse({ ok: false, error: `http_${res.status}` });
        return;
      }
      sendResponse({ ok: true, data: await res.json(), uiCondition });
    } catch (err) {
      sendResponse({
        ok: false,
        error: err.name === "AbortError" ? "timeout" : "unreachable",
      });
    } finally {
      clearTimeout(timer);
    }
  })();

  return true; // keep the message channel open for the async response
});

// ------------------------------------------------------- behavioural events

/**
 * Client-side events the backend log cannot see: whether sources were opened,
 * whether a source was clicked, how long the panel stayed open.
 *
 * Source-opening rate is the primary behavioural measure separating the two
 * UI conditions, so it matters more than the self-report items.
 */
chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type !== "FACTSHIELD_EVENT") return false;
  ensureIdentity().then(({ participantId, uiCondition }) => {
    fetch(`${API_BASE}/v1/event`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Participant-Id": participantId,
        "X-Ui-Condition": uiCondition,
      },
      body: JSON.stringify(msg.event),
    }).catch(() => {}); // telemetry must never break the page
  });
  return false;
});
