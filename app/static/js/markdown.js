(function () {
  function escapeHtml(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
  }

  function inlineFormat(text) {
    let s = escapeHtml(text);
    s = s.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
    s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.*?)__/g, '<strong>$1</strong>');
    s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
    s = s.replace(/_((?!\s).*?(?!\s))_/g, '<em>$1</em>');
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    return s;
  }

  function render(text) {
    if (!text) return '';
    const lines = String(text).split('\n');
    const out = [];
    let inCode = false;
    let codeLang = '';
    let codeBuf = [];
    let inList = null; // 'ul' | 'ol'
    let listBuf = [];

    function flushList() {
      if (!inList) return;
      out.push(`<${inList}>${listBuf.map((li) => `<li>${li}</li>`).join('')}</${inList}>`);
      inList = null;
      listBuf = [];
    }

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i];

      // Fenced code block toggle
      const fence = raw.match(/^```(\w*)/);
      if (!inCode && fence) {
        flushList();
        inCode = true;
        codeLang = escapeHtml(fence[1] || '');
        codeBuf = [];
        continue;
      }
      if (inCode) {
        if (raw.startsWith('```')) {
          const langAttr = codeLang ? ` class="language-${codeLang}"` : '';
          out.push(`<pre><code${langAttr}>${codeBuf.map(escapeHtml).join('\n')}</code></pre>`);
          inCode = false;
          codeLang = '';
          codeBuf = [];
        } else {
          codeBuf.push(raw);
        }
        continue;
      }

      // Heading (h1-h6 map to h4-h6 in bubbles to avoid oversized headings)
      const hm = raw.match(/^(#{1,6})\s+(.*)/);
      if (hm) {
        flushList();
        const level = Math.min(6, hm[1].length + 3);
        out.push(`<h${level}>${inlineFormat(hm[2])}</h${level}>`);
        continue;
      }

      // Horizontal rule
      if (raw.match(/^[-*_]{3,}\s*$/)) {
        flushList();
        out.push('<hr>');
        continue;
      }

      // Unordered list item
      const ulm = raw.match(/^[ \t]*[-*+]\s+(.*)/);
      if (ulm) {
        if (inList === 'ol') flushList();
        inList = 'ul';
        listBuf.push(inlineFormat(ulm[1]));
        continue;
      }

      // Ordered list item
      const olm = raw.match(/^[ \t]*\d+\.\s+(.*)/);
      if (olm) {
        if (inList === 'ul') flushList();
        inList = 'ol';
        listBuf.push(inlineFormat(olm[1]));
        continue;
      }

      // Blank line — flush list and emit paragraph break
      if (raw.trim() === '') {
        flushList();
        out.push('');
        continue;
      }

      // Normal paragraph line
      flushList();
      out.push(`<p>${inlineFormat(raw)}</p>`);
    }

    // Flush unclosed code block
    if (inCode && codeBuf.length) {
      const langAttr = codeLang ? ` class="language-${codeLang}"` : '';
      out.push(`<pre><code${langAttr}>${codeBuf.map(escapeHtml).join('\n')}</code></pre>`);
    }

    flushList();
    return out.filter((l) => l !== '').join('\n');
  }

  window.MarkdownRenderer = { render, escapeHtml };
})();
