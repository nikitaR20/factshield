/**
 * Context capture.
 *
 * This is the part a copy-paste cannot reproduce automatically. When someone
 * highlights "he said it would double by next year", the name is two
 * paragraphs up, "it" is in the previous sentence, and "next year" resolves
 * against the article's publication date. All of it is in the DOM.
 *
 * Context is captured ALWAYS, never conditionally on the selection looking
 * ambiguous — the page date feeds outdated-fact detection even when the
 * sentence is perfectly self-contained.
 */

(() => {
  const MAX_SELECTION = 5000;
  const MAX_CONTEXT = 8000;

  function containingBlock(node) {
    let el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
    while (el && el !== document.body) {
      const display = getComputedStyle(el).display;
      if (display === "block" || display === "list-item" || el.tagName === "P") return el;
      el = el.parentElement;
    }
    return el || document.body;
  }

  /** Two preceding block elements, for pronoun antecedents. */
  function precedingText(block, howMany = 2) {
    const out = [];
    let cur = block;
    for (let i = 0; i < howMany && cur; i++) {
      cur = cur.previousElementSibling;
      if (!cur) break;
      const t = (cur.innerText || "").trim();
      if (t.length > 20) out.unshift(t);
    }
    return out.join("\n\n");
  }

  function meta(...names) {
    for (const n of names) {
      const el =
        document.querySelector(`meta[property="${n}"]`) ||
        document.querySelector(`meta[name="${n}"]`) ||
        document.querySelector(`meta[itemprop="${n}"]`);
      const v = el?.getAttribute("content");
      if (v) return v;
    }
    return null;
  }

  /**
   * Publication date, most reliable source first. JSON-LD and meta tags beat
   * parsing visible text, which is localised and inconsistent.
   */
  function publishedDate() {
    for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const blocks = [].concat(JSON.parse(script.textContent));
        for (const b of blocks) {
          const d = b?.datePublished || b?.dateCreated || b?.["@graph"]?.[0]?.datePublished;
          if (d) return String(d).slice(0, 10);
        }
      } catch {
        /* malformed JSON-LD is common; ignore */
      }
    }
    const m = meta("article:published_time", "datePublished", "publish-date", "date");
    if (m) return String(m).slice(0, 10);

    const t = document.querySelector("time[datetime]")?.getAttribute("datetime");
    return t ? String(t).slice(0, 10) : null;
  }

  function headline() {
    return (
      meta("og:title") ||
      document.querySelector("h1")?.innerText?.trim() ||
      document.title ||
      null
    );
  }

  window.__factshieldCapture = function capture() {
    const sel = window.getSelection();
    const text = (sel?.toString() || "").trim();
    if (!text) return null;

    let surrounding = "";
    try {
      const range = sel.getRangeAt(0);
      const block = containingBlock(range.startContainer);
      const before = precedingText(block);
      const own = (block.innerText || "").trim();
      surrounding = [before, own].filter(Boolean).join("\n\n");
    } catch {
      surrounding = "";
    }

    return {
      selection: text.slice(0, MAX_SELECTION),
      surrounding_text: surrounding.slice(0, MAX_CONTEXT),
      page_title: headline(),
      page_published: publishedDate(),
      page_domain: location.hostname.replace(/^www\./, ""),
      page_lang: document.documentElement.lang || null,
    };
  };

  /** Anchor rect, so the panel appears near the text rather than over it. */
  window.__factshieldAnchor = function anchor() {
    try {
      const r = window.getSelection().getRangeAt(0).getBoundingClientRect();
      return { top: r.bottom + window.scrollY, left: r.left + window.scrollX, width: r.width };
    } catch {
      return { top: window.scrollY + 80, left: window.scrollX + 40, width: 0 };
    }
  };
})();
