/**
 * Background service worker (FactShield 2).
 *
 * Two jobs: the right-click menu / keyboard shortcut, and proxying calls to
 * the local server. The content script cannot call localhost itself without
 * running into the host page's security policy, so every request goes
 * through here. No participant ids, no event logging: there is no user study.
 *
 * MV3 stops this worker when idle. The menu is created in onInstalled, and
 * nothing is kept in memory between events.
 */

const MENU_ID = "factshield-check";
const API_BASE = "http://localhost:8000";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_ID,
      title: "Check with FactShield",
      contexts: ["selection"],
    });
  });
});

function start(tabId) {
  if (tabId) chrome.tabs.sendMessage(tabId, { type: "FACTSHIELD_START" }).catch(() => {});
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === MENU_ID) start(tab?.id);
});

chrome.commands?.onCommand.addListener(async (command) => {
  if (command !== "check-selection") return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  start(tab?.id);
});

/** {type: "FACTSHIELD_API", method, path, body} -> {ok, status, data, error} */
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type !== "FACTSHIELD_API") return false;
  (async () => {
    const controller = new AbortController();
    // Extraction is one model call; starting a check and polling are instant.
    const timer = setTimeout(() => controller.abort(), 120000);
    try {
      const res = await fetch(`${API_BASE}${msg.path}`, {
        method: msg.method || "GET",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: msg.body ? JSON.stringify(msg.body) : undefined,
      });
      let data = null;
      try { data = await res.json(); } catch { data = null; }
      sendResponse({ ok: res.ok, status: res.status, data });
    } catch (err) {
      sendResponse({ ok: false, status: 0, error: err.name === "AbortError" ? "timeout" : "unreachable" });
    } finally {
      clearTimeout(timer);
    }
  })();
  return true; // async response
});
