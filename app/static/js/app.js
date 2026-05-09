// ── DOM refs ──────────────────────────────────────
    const newChatBtn       = document.getElementById('newChatBtn');
    const askBtn           = document.getElementById('askBtn');
    const paperclipBtn     = document.getElementById('paperclipBtn');
    const docFile          = document.getElementById('docFile');
    const attachmentTray   = document.getElementById('attachmentTray');
    const uploadStatusBar  = document.getElementById('uploadStatusBar');
    const inlineError      = document.getElementById('inlineError');
    const question         = document.getElementById('question');
    const chatThread       = document.getElementById('chatThread');
    const sessionsList     = document.getElementById('sessionsList');
    const mainTitle        = document.getElementById('mainTitle');
    const mainMeta         = document.getElementById('mainMeta');
    const clearHistoryBtn  = document.getElementById('clearHistoryBtn');
    const refreshBtn       = document.getElementById('refreshBtn');
    const sidebarToggle    = document.getElementById('sidebarToggle');
    const sidebarBackdrop  = document.getElementById('sidebarBackdrop');
    const apiKeyModal      = document.getElementById('apiKeyModal');
    const modalTitle       = document.getElementById('modalTitle');
    const modalSubtitle    = document.getElementById('modalSubtitle');
    const modalApiKeyInput = document.getElementById('modalApiKeyInput');
    const modalError       = document.getElementById('modalError');
    const modalCancelBtn   = document.getElementById('modalCancelBtn');
    const modalSubmitBtn   = document.getElementById('modalSubmitBtn');
    const modelsBtn        = document.getElementById('modelsBtn');
    const modelPanel       = document.getElementById('modelPanel');
    const modelsBtnLabel   = document.getElementById('modelsBtnLabel');
    const composerMeta     = document.getElementById('composerMeta');
    const darkToggleBtn    = document.getElementById('darkToggleBtn');
    // Legacy aliases kept for compatibility within this file
    const attachBtn        = paperclipBtn;
    const uploadStatus     = uploadStatusBar;
    const attachedDocs     = attachmentTray;

    // ── State ─────────────────────────────────────────
    let currentChatId = null;
    let currentModel  = '';          // replaces the hidden <select>
    let cachedSessions = [];
    let historyRequestId = 0;
    let historyController = null;
    let uploadPending = false;
    let chatPending = false;
    let suppressThreadRender = false;
    let defaultModel = '';
    let draftAttachments = [];
    const attachmentsByChat = new Map();

    // External provider state
    let activeProvider = 'local'; // "local" | "chatgpt" | "gemini" | "claude"
    let externalApiKeys = { chatgpt: null, gemini: null, claude: null };
    const SESS_KEY_PREFIX = 'mira_apikey_';

    // Restore API keys saved in this browser session
    (['chatgpt', 'gemini', 'claude']).forEach((p) => {
      const stored = sessionStorage.getItem(SESS_KEY_PREFIX + p);
      if (stored) externalApiKeys[p] = stored;
    });
    let externalModels = { chatgpt: [], gemini: [], claude: [] };
    let ollamaModels = []; // Cached local model list
    let pendingProvider = null; // provider being validated in modal
    let pendingModel = null;    // model being validated in modal

    function providerLabel(provider) {
      if (provider === 'chatgpt') return 'ChatGPT';
      if (provider === 'gemini') return 'Gemini';
      if (provider === 'claude') return 'Claude';
      return 'Local';
    }

    function setSidebarOpen(open) {
      document.body.classList.toggle('sidebar-open', !!open);
    }

    function updateComposerMeta() {
      const model = currentModel || defaultModel || 'Not selected';
      const provider = providerLabel(activeProvider);
      const keyStatus = activeProvider === 'local'
        ? 'No API key required'
        : (externalApiKeys[activeProvider] ? 'API key verified' : 'API key needed');
      composerMeta.textContent = `Provider: ${provider} · Model: ${model} · ${keyStatus}`;

      if (modelsBtnLabel) {
        modelsBtnLabel.textContent = model.length > 18 ? model.slice(0, 17) + '…' : model;
        modelsBtn.title = model;
      }
    }

    // ── Model Panel ───────────────────────────────────
    const PROVIDER_META = {
      local: {
        label: 'Local',
        icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><polyline points="8 21 12 17 16 21"/></svg>`,
      },
      chatgpt: {
        label: 'ChatGPT',
        icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M22.282 9.821a5.985 5.985 0 00-.516-4.91 6.046 6.046 0 00-6.51-2.9A6.065 6.065 0 004.981 4.18a5.985 5.985 0 00-3.998 2.9 6.046 6.046 0 00.743 7.097 5.98 5.98 0 00.51 4.911 6.051 6.051 0 006.515 2.9A5.985 5.985 0 0013.26 24a6.056 6.056 0 005.772-4.206 5.99 5.99 0 003.997-2.9 6.056 6.056 0 00-.747-7.073zM13.26 22.43a4.476 4.476 0 01-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 00.392-.681v-6.737l2.02 1.168a.071.071 0 01.038.052v5.583a4.504 4.504 0 01-4.494 4.494zM3.6 18.304a4.47 4.47 0 01-.535-3.014l.142.085 4.783 2.759a.771.771 0 00.78 0l5.843-3.369v2.332a.08.08 0 01-.033.062L9.74 19.95a4.5 4.5 0 01-6.14-1.646zM2.34 7.896a4.485 4.485 0 012.366-1.973V11.6a.766.766 0 00.388.676l5.815 3.355-2.02 1.168a.076.076 0 01-.071 0l-4.83-2.786A4.504 4.504 0 012.34 7.896zm16.597 3.855l-5.843-3.369 2.02-1.168a.076.076 0 01.071 0l4.83 2.786a4.494 4.494 0 01-.676 8.105V12.57a.79.79 0 00-.402-.82zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 00-.785 0L9.409 9.23V6.897a.066.066 0 01.028-.061l4.83-2.787a4.5 4.5 0 016.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 01-.038-.057V6.075a4.5 4.5 0 017.375-3.453l-.142.08L8.704 5.46a.795.795 0 00-.393.681zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5z"/></svg>`,
      },
      gemini: {
        label: 'Gemini',
        icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 1.8l2.1 5.1L19.2 9l-5.1 2.1L12 16.2l-2.1-5.1L4.8 9l5.1-2.1L12 1.8z"/><path d="M18.4 14.4l1 2.4 2.4 1-2.4 1-1 2.4-1-2.4-2.4-1 2.4-1 1-2.4z"/><path d="M6.1 14.7l.75 1.8 1.8.75-1.8.75-.75 1.8-.75-1.8-1.8-.75 1.8-.75.75-1.8z"/></svg>`,
      },
      claude: {
        label: 'Claude',
        icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7.6 20H4.3L10 4h4l5.7 16h-3.3l-1.2-3.6H8.8L7.6 20zm2.1-6.1h4.6L12 7.1l-2.3 6.8z"/></svg>`,
      },
    };

    let panelActiveProv = null; // which provider is highlighted in the left column

    // Known release order per provider — newest first.
    // Unknown models (fetched live) fall to the bottom in their original order.
    const MODEL_RELEASE_ORDER = {
      chatgpt: [
        'gpt-4.1', 'gpt-4.1-mini',
        'gpt-4o', 'gpt-4o-mini',
        'gpt-4-turbo', 'gpt-4',
        'gpt-3.5-turbo',
      ],
      gemini: [
        'gemini-2.5-pro', 'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-1.5-pro', 'gemini-1.5-flash',
      ],
      claude: [
        'claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-3-5',
        'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
      ],
    };

    function sortModelsByRecency(provider, models) {
      const order = MODEL_RELEASE_ORDER[provider] || [];
      if (!order.length) return models;
      return [...models].sort((a, b) => {
        const aLow = a.toLowerCase();
        const bLow = b.toLowerCase();
        const aRank = (() => { const i = order.findIndex(k => aLow.startsWith(k)); return i === -1 ? order.length : i; })();
        const bRank = (() => { const i = order.findIndex(k => bLow.startsWith(k)); return i === -1 ? order.length : i; })();
        return aRank - bRank;
      });
    }

    function renderModelPanelRight(prov) {
      panelActiveProv = prov;
      // Highlight active row in providers column
      modelPanel.querySelectorAll('.mp-provider-row').forEach((r) => {
        r.classList.toggle('active', r.dataset.provider === prov);
      });
      const modelsCol = modelPanel.querySelector('.mp-models');
      if (!modelsCol) return;
      const rawModels = prov === 'local'
        ? ollamaModels.map((m) => m.name || m)
        : (externalModels[prov] || []).map((m) => m.name || m);
      const models = prov === 'local' ? rawModels : sortModelsByRecency(prov, rawModels);
      if (!models.length) {
        modelsCol.innerHTML = `<div class="mp-models-empty">No models loaded</div>`;
        return;
      }
      modelsCol.innerHTML = models.map((m) => {
        const isActive = m === currentModel && prov === activeProvider;
        return `<div class="model-item${isActive ? ' active' : ''}" data-provider="${escapeHtml(prov)}" data-model="${escapeHtml(m)}">
          <span class="model-check">${isActive ? '✓' : ''}</span>
          <span class="model-name-text" title="${escapeHtml(m)}">${escapeHtml(m)}</span>
        </div>`;
      }).join('');
      modelsCol.querySelectorAll('.model-item[data-model]').forEach((item) => {
        item.addEventListener('click', () => pickModel(item.dataset.provider, item.dataset.model));
      });
    }

    function renderModelPanel() {
      const providers = ['local', 'chatgpt', 'gemini', 'claude'];
      const initProv = activeProvider;
      const leftHtml = providers.map((prov) => {
        const { icon, label } = PROVIDER_META[prov];
        const hasKey = prov === 'local' || !!externalApiKeys[prov];
        const badgeHtml = prov === 'local'
          ? `<span class="ms-badge">No key</span>`
          : (hasKey ? `<span class="ms-badge">Key ✓</span>` : `<span class="ms-badge warn">Key?</span>`);
        return `<div class="mp-provider-row${prov === initProv ? ' active' : ''}" data-provider="${escapeHtml(prov)}">
          <span class="ms-icon" style="display:inline-flex;align-items:center;">${icon}</span>
          <span class="ms-name">${label}</span>
          ${badgeHtml}
        </div>`;
      }).join('');

      modelPanel.innerHTML = `<div class="mp-models"></div><div class="mp-providers">${leftHtml}</div>`;

      modelPanel.querySelectorAll('.mp-provider-row').forEach((row) => {
        row.addEventListener('click', () => renderModelPanelRight(row.dataset.provider));
      });

      renderModelPanelRight(initProv);
    }

    function openPanel() {
      renderModelPanel();
      modelPanel.classList.add('open');
      positionModelPanel();
      modelsBtn.classList.add('open');
      modelsBtn.setAttribute('aria-expanded', 'true');
    }

    function closePanel() {
      modelPanel.classList.remove('open');
      modelsBtn.classList.remove('open');
      modelsBtn.setAttribute('aria-expanded', 'false');
    }

    function positionModelPanel() {
      if (!modelPanel.classList.contains('open')) return;

      const viewportMargin = 8;
      const panelGap = 8;
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const btnRect = modelsBtn.getBoundingClientRect();

      // Reset any previously forced dimensions so CSS fit-content applies
      modelPanel.style.width = '';
      modelPanel.style.maxWidth = `${viewportWidth - viewportMargin * 2}px`;
      modelPanel.style.maxHeight = `${viewportHeight - viewportMargin * 2}px`;

      const panelWidth = modelPanel.offsetWidth;
      const openLeftPreferred = (btnRect.left + btnRect.width / 2) > (viewportWidth / 2);

      let left = openLeftPreferred ? (btnRect.right - panelWidth) : btnRect.left;
      left = Math.max(viewportMargin, Math.min(left, viewportWidth - panelWidth - viewportMargin));

      const panelHeight = modelPanel.offsetHeight;
      const spaceBelow = viewportHeight - btnRect.bottom - panelGap;
      const spaceAbove = btnRect.top - panelGap;
      const openUpward = spaceBelow < panelHeight && spaceAbove > spaceBelow;

      let top = openUpward ? (btnRect.top - panelGap - panelHeight) : (btnRect.bottom + panelGap);
      top = Math.max(viewportMargin, Math.min(top, viewportHeight - panelHeight - viewportMargin));

      modelPanel.style.left = `${left}px`;
      modelPanel.style.top = `${top}px`;
    }

    modelsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (modelPanel.classList.contains('open')) closePanel();
      else openPanel();
    });

    document.addEventListener('click', (e) => {
      if (!modelPanel.contains(e.target) && e.target !== modelsBtn) closePanel();
    });

    window.addEventListener('resize', () => {
      if (modelPanel.classList.contains('open')) positionModelPanel();
    });

    window.addEventListener('scroll', () => {
      if (modelPanel.classList.contains('open')) positionModelPanel();
    }, { passive: true });

    // ── Keyboard shortcuts ────────────────────────────
    document.addEventListener('keydown', (e) => {
      const mod = e.metaKey || e.ctrlKey;
      if (e.key === 'Escape') {
        closePanel();
        apiKeyModal.classList.remove('open');
        closeCtxMenu();
        return;
      }
      if (mod && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (modelPanel.classList.contains('open')) closePanel();
        else openPanel();
        return;
      }
      if (mod && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        newChatBtn.click();
        question.focus();
      }
    });

    function pickModel(provider, modelName) {
      if (provider !== 'local' && !externalApiKeys[provider]) {
        // Need API key — show modal, then on success apply
        pendingProvider = provider;
        pendingModel = modelName;
        modalTitle.textContent = `Enter ${providerLabel(provider)} API Key`;
        modalSubtitle.textContent = 'Your key is used only for this session and never stored.';
        modalApiKeyInput.value = '';
        modalError.textContent = '';
        apiKeyModal.classList.add('open');
        modalApiKeyInput.focus();
        closePanel();
      } else {
        applyModel(provider, modelName);
        closePanel();
      }
    }

    function applyModel(provider, modelName) {
      applyProvider(provider);
      currentModel = modelName;
      activeProvider = provider;
      updateComposerMeta();
    }

    function sleep(ms) {
      return new Promise((resolve) => window.setTimeout(resolve, ms));
    }

    // ── Inline error helper ───────────────────────────
    let inlineErrorTimer = null;
    function showInlineError(msg, durationMs = 4500) {
      inlineError.textContent = msg;
      inlineError.classList.add('show');
      if (inlineErrorTimer) clearTimeout(inlineErrorTimer);
      inlineErrorTimer = setTimeout(() => inlineError.classList.remove('show'), durationMs);
    }

    // ── Dark mode ─────────────────────────────────────
    const DARK_KEY = 'phi-rag-dark';
    function applyDark(dark) {
      document.body.classList.toggle('dark', dark);
      darkToggleBtn.textContent = dark ? '☀️' : '🌙';
    }
    applyDark(localStorage.getItem(DARK_KEY) === '1');
    darkToggleBtn.addEventListener('click', () => {
      const next = !document.body.classList.contains('dark');
      applyDark(next);
      localStorage.setItem(DARK_KEY, next ? '1' : '0');
    });

    paperclipBtn.addEventListener('click', () => docFile.click());

    // ── Utilities ─────────────────────────────────────
    function setBusy(btn, busy, busyLabel, idleLabel) {
      btn.disabled = busy;
      btn.textContent = busy ? busyLabel : idleLabel;
    }

    function escapeHtml(v) {
      return String(v)
        .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;').replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    async function fetchJson(url, options = {}, timeoutMs = 15000) {
      const ctrl = new AbortController();
      const tid = window.setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const res = await fetch(url, { ...options, signal: ctrl.signal });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        return data;
      } catch (err) {
        if (err.name === 'AbortError') throw new Error('Request timed out.');
        throw err;
      } finally {
        window.clearTimeout(tid);
      }
    }

    async function streamChat(payload, onChunk, timeoutMs = 480000) {
      const ctrl = new AbortController();
      const tid = window.setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const res = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: ctrl.signal,
        });

        if (!res.ok) {
          let detail = 'Request failed';
          try {
            const errData = await res.json();
            detail = errData.detail || detail;
          } catch (_) {}
          throw new Error(detail);
        }

        if (!res.body) throw new Error('Streaming is not supported in this browser.');

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalEvent = null;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          while (true) {
            const sep = buffer.indexOf('\n\n');
            if (sep === -1) break;
            const rawEvent = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);

            const lines = rawEvent.split('\n');
            for (const line of lines) {
              if (!line.startsWith('data:')) continue;
              const jsonText = line.slice(5).trim();
              if (!jsonText) continue;

              let evt;
              try {
                evt = JSON.parse(jsonText);
              } catch (_) {
                continue;
              }

              if (evt.chunk) onChunk(String(evt.chunk));
              if (evt.done) finalEvent = evt;
            }
          }
        }

        if (!finalEvent) throw new Error('Streaming ended unexpectedly.');
        return finalEvent;
      } catch (err) {
        if (err.name === 'AbortError') throw new Error('Request timed out.');
        throw err;
      } finally {
        window.clearTimeout(tid);
      }
    }

    function timeAgo(isoStr) {
      const diff = Date.now() - new Date(isoStr).getTime();
      if (isNaN(diff) || diff < 0) return '';
      const m = Math.floor(diff / 60000);
      if (m < 1) return 'just now';
      if (m < 60) return m + 'm ago';
      const h = Math.floor(m / 60);
      if (h < 24) return 'Today';
      if (h < 48) return 'Yesterday';
      const d = Math.floor(h / 24);
      if (d < 7) return d + ' days ago';
      if (d < 14) return 'Last week';
      if (d < 60) return Math.floor(d / 7) + ' weeks ago';
      return 'Last month';
    }

    function renderModelOptions(models, selected) {
      const safeModels = Array.isArray(models) ? models : [];
      currentModel = selected || safeModels[0]?.name || safeModels[0] || '';
    }

    function applyProvider(provider) {
      activeProvider = provider;
      if (provider === 'local') {
        renderModelOptions(ollamaModels, defaultModel);
        updateComposerMeta();
      } else {
        const models = externalModels[provider] || [];
        renderModelOptions(models.map((m) => ({ name: m.name || m, installed: true })), models[0]?.name || models[0] || '');
        updateComposerMeta();
      }
    }

    // Modal cancel
    modalCancelBtn.addEventListener('click', () => {
      apiKeyModal.classList.remove('open');
      pendingProvider = null;
    });
    apiKeyModal.addEventListener('click', (e) => {
      if (e.target === apiKeyModal) {
        apiKeyModal.classList.remove('open');
        pendingProvider = null;
      }
    });

    // Focus trap — keep Tab/Shift+Tab inside the modal while it is open
    apiKeyModal.addEventListener('keydown', (e) => {
      if (e.key !== 'Tab') return;
      const focusable = Array.from(
        apiKeyModal.querySelectorAll('button:not([disabled]), input, [tabindex]:not([tabindex="-1"])')
      );
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey) {
        if (document.activeElement === first) { e.preventDefault(); last.focus(); }
      } else {
        if (document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });

    // Modal submit
    modalSubmitBtn.addEventListener('click', async () => {
      const apiKey = modalApiKeyInput.value.trim();
      if (!apiKey) { modalError.textContent = 'Please enter an API key.'; return; }

      modalError.textContent = '';
      modalSubmitBtn.disabled = true;
      modalSubmitBtn.textContent = 'Validating…';

      const provider = pendingProvider;
      const model = (externalModels[provider]?.[0]?.name) || (externalModels[provider]?.[0]) || '';

      try {
        const result = await fetchJson('/api-key/validate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider, model, api_key: apiKey }),
        }, 20000);

        if (result.valid) {
          externalApiKeys[provider] = apiKey;
          sessionStorage.setItem(SESS_KEY_PREFIX + provider, apiKey);
          if (Array.isArray(result.models) && result.models.length) {
            externalModels[provider] = result.models;
          }
          apiKeyModal.classList.remove('open');
          pendingProvider = null;
          pendingModel = null;
          applyProvider(provider);
          updateComposerMeta();
          openPanel();
        } else {
          modalError.textContent = result.error || 'Invalid API key. Please try again.';
        }
      } catch (err) {
        modalError.textContent = 'Validation failed: ' + err.message;
      } finally {
        modalSubmitBtn.disabled = false;
        modalSubmitBtn.textContent = 'Validate & Save';
      }
    });

    modalApiKeyInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') modalSubmitBtn.click();
    });

    async function loadModelOptions() {
      try {
        const payload = await fetchJson('/models', {}, 12000);
        const models = payload.models || [];
        defaultModel = payload.default_model || '';
        ollamaModels = models; // Cache for instant switching
        // Store external model lists
        const ext = payload.external_providers || {};
        externalModels.chatgpt = (ext.chatgpt?.models || []);
        externalModels.gemini = (ext.gemini?.models || []);
        externalModels.claude = (ext.claude?.models || []);
        if (activeProvider === 'local') {
          renderModelOptions(models, defaultModel);
        } else {
          const providerModels = externalModels[activeProvider] || [];
          renderModelOptions(
            providerModels.map((m) => ({ name: m.name || m, installed: true })),
            providerModels[0]?.name || providerModels[0] || ''
          );
        }
        updateComposerMeta();
      } catch (err) {
        if (activeProvider === 'local') {
          renderModelOptions([
            { name: 'phi4-mini:3.8b-q4_K_M', installed: true },
            { name: 'qwen2.5:3b-instruct-q4_K_M', installed: true },
            { name: 'qwen2.5:1.5b', installed: true },
          ], 'phi4-mini:3.8b-q4_K_M');
        }
        updateComposerMeta();
      }
    }

    sidebarToggle.addEventListener('click', () => {
      const isOpen = document.body.classList.contains('sidebar-open');
      setSidebarOpen(!isOpen);
    });
    sidebarBackdrop.addEventListener('click', () => setSidebarOpen(false));

    function getActiveAttachments() {
      if (!currentChatId) return draftAttachments;
      return attachmentsByChat.get(currentChatId) || [];
    }

    function setActiveAttachments(nextAttachments) {
      const normalized = Array.isArray(nextAttachments)
        ? nextAttachments.map((x) => String(x || '').trim()).filter(Boolean)
        : [];
      if (!currentChatId) {
        draftAttachments = normalized;
        return;
      }
      attachmentsByChat.set(currentChatId, normalized);
    }

    function renderAttachments() {
      const items = getActiveAttachments();
      attachmentTray.classList.toggle('has-items', items.length > 0);
      if (!items.length) {
        attachmentTray.innerHTML = '';
        return;
      }
      attachmentTray.innerHTML = items
        .map((name, idx) => {
          const safe = escapeHtml(name);
          return `<span class="attached-chip">📎 <span class="attached-chip-name">${safe}</span><button class="attached-chip-remove" data-idx="${idx}" title="Remove attachment">✕</button></span>`;
        })
        .join('');
      attachmentTray.querySelectorAll('.attached-chip-remove').forEach((btn) => {
        btn.addEventListener('click', () => {
          const i = Number(btn.dataset.idx);
          const next = getActiveAttachments().slice();
          next.splice(i, 1);
          setActiveAttachments(next);
          renderAttachments();
        });
      });
    }

    function mergeSessionAttachments(sessions) {
      const safeSessions = Array.isArray(sessions) ? sessions : [];
      safeSessions.forEach((session) => {
        const sid = String(session.id || '');
        if (!sid) return;
        const sources = [];
        const turns = Array.isArray(session.turns) ? session.turns : [];
        turns.forEach((turn) => {
          const tSources = Array.isArray(turn.sources) ? turn.sources : [];
          tSources.forEach((src) => {
            const s = String(src || '').trim();
            if (s && !sources.includes(s)) sources.push(s);
          });
        });
        attachmentsByChat.set(sid, sources);
      });
    }

    // ── Auto-resize textarea + Enter to send ──────────
    question.addEventListener('input', () => {
      question.style.height = 'auto';
      question.style.height = Math.min(question.scrollHeight, 140) + 'px';
    });
    question.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askBtn.click(); }
    });

    // ── Session sidebar ───────────────────────────────
    let openCtxId = null;

    function closeCtxMenu() {
      if (openCtxId) {
        const encoded = encodeURIComponent(openCtxId);
        const m = sessionsList.querySelector(`.session-ctx-menu[data-sid="${CSS.escape(encoded)}"]`);
        if (m) m.classList.remove('open');
        openCtxId = null;
      }
    }

    function renderSessions(sessions) {
      cachedSessions = sessions;
      if (!sessions.length) {
        sessionsList.innerHTML = '<div class="sessions-empty">No chats yet.</div>';
        return;
      }
      sessionsList.innerHTML = sessions.map((s, idx) => {
        const title = escapeHtml(s.title || 'Untitled Chat');
        const active = s.id === currentChatId ? ' active' : '';
        const sid = encodeURIComponent(String(s.id || ''));
        const dividerClass = idx > 0 ? ' with-divider' : '';
        return `<div class="session-tab${active}${dividerClass}" data-sid="${sid}">
          <div class="session-tab-row" data-action="select-session">
            <span class="session-bubble-icon">💬</span>
            <div class="session-tab-body">
              <div class="session-tab-title">${title}</div>
              <div class="session-tab-time">${timeAgo(s.updated_at)}</div>
            </div>
            <button class="session-tab-more" data-action="open-menu" title="More options">•••</button>
          </div>
          <div class="session-ctx-menu" data-sid="${sid}">
            <div class="ctx-item" data-action="rename-session">✏️ Rename</div>
            <div class="ctx-item danger" data-action="delete-session">🗑 Delete</div>
          </div>
        </div>`;
      }).join('');

      sessionsList.querySelectorAll('.session-tab-row[data-action="select-session"]').forEach((row) => {
        row.addEventListener('click', () => {
          const tab = row.closest('.session-tab');
          const sid = decodeURIComponent(tab?.dataset?.sid || '');
          if (sid) selectSession(sid);
        });
      });

      sessionsList.querySelectorAll('.session-tab-more[data-action="open-menu"]').forEach((btn) => {
        btn.addEventListener('click', (event) => {
          event.stopPropagation();
          const tab = btn.closest('.session-tab');
          const sid = decodeURIComponent(tab?.dataset?.sid || '');
          if (sid) toggleCtxMenu(sid);
        });
      });

      sessionsList.querySelectorAll('.ctx-item[data-action="rename-session"]').forEach((item) => {
        item.addEventListener('click', () => {
          const tab = item.closest('.session-tab');
          const sid = decodeURIComponent(tab?.dataset?.sid || '');
          if (sid) renameSession(sid);
        });
      });

      sessionsList.querySelectorAll('.ctx-item[data-action="delete-session"]').forEach((item) => {
        item.addEventListener('click', () => {
          const tab = item.closest('.session-tab');
          const sid = decodeURIComponent(tab?.dataset?.sid || '');
          if (sid) deleteSession(sid);
        });
      });
    }

    function toggleCtxMenu(sessionId) {
      if (openCtxId === sessionId) { closeCtxMenu(); return; }
      closeCtxMenu();
      const encoded = encodeURIComponent(sessionId);
      const m = sessionsList.querySelector(`.session-ctx-menu[data-sid="${CSS.escape(encoded)}"]`);
      if (m) { m.classList.add('open'); openCtxId = sessionId; }
    }

    async function renameSession(sessionId) {
      closeCtxMenu();
      const s = cachedSessions.find((x) => x.id === sessionId);
      const current = s ? s.title : '';
      const newTitle = prompt('Rename chat:', current);
      if (!newTitle || newTitle === current) return;
      const trimmed = newTitle.trim();
      if (!trimmed) return;

      const payload = await fetchJson(`/chat/${encodeURIComponent(sessionId)}/title`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: trimmed }),
      });

      if (s) s.title = payload.title || trimmed;
      if (currentChatId === sessionId) {
        mainTitle.textContent = payload.title || trimmed;
      }
      renderSessions(cachedSessions);
    }

    async function deleteSession(sessionId) {
      closeCtxMenu();
      if (!confirm('Delete this chat?')) return;
      await fetchJson(`/chat/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
      // Remove from cache & re-render
      cachedSessions = cachedSessions.filter((x) => x.id !== sessionId);
      attachmentsByChat.delete(sessionId);
      if (currentChatId === sessionId) {
        currentChatId = null;
        mainTitle.textContent = 'New Chat';
        mainMeta.textContent = '';
        chatThread.innerHTML = `<div class="chat-empty">
          <div class="empty-illustration">✦</div>
          <div class="chat-empty-title">Chat deleted</div>
          <div class="chat-empty-hint">Start a fresh conversation any time.</div>
        </div>`;
      }
      renderSessions(cachedSessions);
    }

    document.addEventListener('click', (e) => {
      if (openCtxId && !e.target.closest('.session-ctx-menu') && !e.target.closest('.session-tab-more')) {
        closeCtxMenu();
      }
    });

    function syncActiveSidebarTab() {
      document.querySelectorAll('.session-tab').forEach((el) => {
        el.classList.toggle('active', el.dataset.sid === currentChatId);
      });
    }

    // ── Chat thread rendering ─────────────────────────
    function turnHtml(t) {
      const sources = (t.sources || [])
        .map((s) => `<span class="source-tag">${escapeHtml(s)}</span>`).join('');
      const ms = t.model_response_ms ? `${escapeHtml(t.model||'')} · ${t.model_response_ms} ms` : '';
      const meta = (sources || ms)
        ? `<div class="bubble-meta">${sources}<span>${escapeHtml(ms)}</span></div>` : '';
      const answerHtml = window.MarkdownRenderer
        ? window.MarkdownRenderer.render(t.answer || '')
        : escapeHtml(t.answer || '');
      return `<div class="turn">
        <div class="bubble-q">${escapeHtml(t.question || '')}</div>
        <div class="bubble-a">${answerHtml}</div>
        ${meta}
      </div>`;
    }

    function renderThread(session) {
      if (!session || !session.turns || !session.turns.length) {
        chatThread.innerHTML = `<div class="chat-empty">
          <div class="empty-illustration">✦</div>
          <div class="chat-empty-title">Session ready</div>
          <div class="chat-empty-hint">Ask your first question, and answers will come only from your indexed documents.</div>
        </div>`;
        return;
      }
      chatThread.innerHTML = session.turns.map(turnHtml).join('');
      chatThread.scrollTop = chatThread.scrollHeight;
    }

    function appendThinkingBubble(q) {
      chatThread.querySelector('.chat-empty')?.remove();
      const div = document.createElement('div');
      div.className = 'turn';
      div.id = 'thinking-turn';
      div.innerHTML = `<div class="bubble-q">${escapeHtml(q)}</div>
        <div class="bubble-a thinking" id="thinking-bubble">
          <span class="dots-wrap"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>
        </div>`;
      chatThread.appendChild(div);
      chatThread.scrollTop = chatThread.scrollHeight;
    }

    function resolveThinkingBubble(turn) {
      const el = document.getElementById('thinking-turn');
      if (el) el.outerHTML = turnHtml(turn);
      chatThread.scrollTop = chatThread.scrollHeight;
    }

    function failThinkingBubble(msg) {
      const el = document.getElementById('thinking-bubble');
      if (el) { el.textContent = 'Error: ' + msg; el.className = 'bubble-a error'; }
    }

    // ── Load history ──────────────────────────────────
    async function loadHistory({ background = false } = {}) {
      const reqId = ++historyRequestId;
      if (historyController) historyController.abort();
      historyController = new AbortController();

      try {
        const res = await fetch('/chat/history?limit_chats=20&limit_turns=50', { signal: historyController.signal });
        const data = await res.json();
        if (!res.ok || reqId !== historyRequestId) return;

        const sessions = data.sessions || [];
        renderSessions(sessions);
        mergeSessionAttachments(sessions);

        if (currentChatId) {
          const s = sessions.find((x) => x.id === currentChatId);
          if (s) {
            mainTitle.textContent = s.title || 'Untitled Chat';
            if (!suppressThreadRender) renderThread(s);
            renderAttachments();
          }
        }
      } catch (err) {
        if (err.name === 'AbortError') return;
        if (!background) sessionsList.innerHTML = `<div class="sessions-empty">Error: ${escapeHtml(err.message)}</div>`;
      } finally {
        if (reqId === historyRequestId) historyController = null;
        suppressThreadRender = false;
      }
    }

    // ── Select session ────────────────────────────────
    function selectSession(sessionId) {
      currentChatId = sessionId;
      syncActiveSidebarTab();
      setSidebarOpen(false);
      const s = cachedSessions.find((x) => x.id === sessionId);
      if (s) {
        mainTitle.textContent = s.title || 'Untitled Chat';
        mainMeta.textContent = '';
        // Restore the provider + model that was used in this session
        if (s.provider && s.model) {
          const restoredProvider = s.provider === 'ollama' ? 'local' : s.provider;
          applyModel(restoredProvider, s.model);
        }
        renderThread(s);
        renderAttachments();
      }
    }

    // ── New Chat ──────────────────────────────────────
    newChatBtn.addEventListener('click', () => {
      currentChatId = null;
      setSidebarOpen(false);
      syncActiveSidebarTab();
      mainTitle.textContent = 'New Chat';
      mainMeta.textContent = '';
      question.value = '';
      question.style.height = 'auto';
      draftAttachments = [];
      uploadStatus.textContent = '';
      uploadStatus.className = 'upload-status';
      renderAttachments();
      chatThread.innerHTML = `<div class="chat-empty">
        <div class="empty-illustration">✦</div>
        <div class="chat-empty-title">New chat</div>
        <div class="chat-empty-hint">Attach one or more files, choose a model, then ask your first question.</div>
      </div>`;
    });

    // ── Upload ────────────────────────────────────────
    docFile.addEventListener('change', async () => {
      if (uploadPending) return;
      if (!docFile.files.length) {
        uploadStatusBar.className = 'upload-status err';
        uploadStatusBar.textContent = 'Select a file first.';
        return;
      }
      if (!currentChatId) {
        currentChatId = (window.crypto && typeof window.crypto.randomUUID === 'function')
          ? window.crypto.randomUUID()
          : `chat-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
        mainTitle.textContent = 'New Chat';
        syncActiveSidebarTab();
      }
      const fd = new FormData();
      fd.append('file', docFile.files[0]);
      fd.append('chat_id', currentChatId);
      uploadStatusBar.className = 'upload-status';
      uploadStatusBar.textContent = 'Uploading and indexing...';
      uploadPending = true;
      paperclipBtn.disabled = true;
      try {
        const d = await fetchJson('/upload', { method: 'POST', body: fd }, 300000);
        const reuse = d.reused_existing_index ? ' (reused existing index)' : '';

        if (d.reused_existing_index) {
          uploadStatusBar.className = 'upload-status ok';
          uploadStatusBar.textContent = `Attached ${d.file}${reuse}`;
          const next = getActiveAttachments().slice();
          if (d.file && !next.includes(d.file)) next.push(d.file);
          setActiveAttachments(next);
          renderAttachments();
          return;
        }

        if (!d.job_id) {
          throw new Error('Upload job was not created.');
        }

        uploadStatusBar.textContent = 'Upload saved. Waiting for indexing job…';

        const MAX_POLL_ATTEMPTS = 150; // 5 min at 2 s intervals
        let pollAttempts = 0;
        while (true) {
          if (++pollAttempts > MAX_POLL_ATTEMPTS) {
            throw new Error('Indexing timed out. The server may still be processing — try refreshing.');
          }
          await sleep(2000);
          const job = await fetchJson(`/upload/jobs/${d.job_id}`, {}, 15000);

          if (job.status === 'queued' || job.status === 'running') {
            uploadStatusBar.className = 'upload-status';
            uploadStatusBar.textContent = job.message || `Indexing ${d.file}…`;
            continue;
          }

          if (job.status === 'failed') {
            throw new Error(job.message || 'Indexing failed.');
          }

          if (job.status === 'completed') {
            const result = job.result || {};
            const doneReuse = result.reused_existing_index ? ' (reused existing index)' : '';
            uploadStatusBar.className = 'upload-status ok';
            uploadStatusBar.textContent = `Attached ${result.file || d.file}${doneReuse}`;
            const attachedName = result.file || d.file || '';
            const next = getActiveAttachments().slice();
            if (attachedName && !next.includes(attachedName)) next.push(attachedName);
            setActiveAttachments(next);
            renderAttachments();
            break;
          }
        }
      } catch (err) {
        uploadStatusBar.className = 'upload-status err';
        uploadStatusBar.textContent = '✗ ' + err.message;
      } finally {
        uploadPending = false;
        paperclipBtn.disabled = false;
        docFile.value = '';
      }
    });

    // ── Ask ───────────────────────────────────────────
    askBtn.addEventListener('click', async () => {
      if (chatPending) return;
      const q = question.value.trim();
      if (!q) return;

      // Guard: no document attached
      if (!getActiveAttachments().length) {
        showInlineError('📎 Please attach a document before asking a question.');
        return;
      }

      // Guard: external provider without API key
      if (activeProvider !== 'local' && !externalApiKeys[activeProvider]) {
        showInlineError(`🔑 API key required for ${providerLabel(activeProvider)}. Click the model button to add it.`);
        modelsBtn.classList.add('key-error');
        setTimeout(() => modelsBtn.classList.remove('key-error'), 1200);
        return;
      }

      inlineError.classList.remove('show');
      appendThinkingBubble(q);
      question.value = '';
      question.style.height = 'auto';
      chatPending = true;
      askBtn.disabled = true;
      askBtn.textContent = 'Thinking…';
      question.classList.add('loading');

      const thinkingBubble = document.getElementById('thinking-bubble');
      let streamedText = '';

      const renderStreamingText = () => {
        if (!thinkingBubble) return;
        thinkingBubble.classList.remove('thinking');
        thinkingBubble.innerHTML = `${escapeHtml(streamedText)}<span class="stream-cursor">|</span>`;
        chatThread.scrollTop = chatThread.scrollHeight;
      };

      try {
        const doneEvent = await streamChat(
          {
            question: q,
            chat_id: currentChatId,
            model: currentModel || defaultModel,
            source_filters: getActiveAttachments(),
            provider: activeProvider,
            api_key: activeProvider !== 'local' ? (externalApiKeys[activeProvider] || null) : null,
          },
          (chunk) => {
            streamedText += chunk;
            renderStreamingText();
          },
          480000,
        );

        if (doneEvent.error && !streamedText.trim()) {
          throw new Error(String(doneEvent.error));
        }

        const oldChatId = currentChatId;
        currentChatId = doneEvent.chat_id || currentChatId;
        if (!oldChatId && currentChatId && draftAttachments.length) {
          attachmentsByChat.set(currentChatId, draftAttachments.slice());
          draftAttachments = [];
        }

        if (doneEvent.model && doneEvent.model_response_ms) {
          mainMeta.textContent = `${doneEvent.model} · ${doneEvent.model_response_ms} ms`;
        }

        const finalAnswer = streamedText || 'No answer generated.';
        resolveThinkingBubble({
          question: q,
          answer: finalAnswer,
          sources: doneEvent.sources || [],
          model: doneEvent.model || (currentModel || defaultModel),
          model_response_ms: doneEvent.model_response_ms || 0,
        });

        suppressThreadRender = true;
        loadHistory({ background: true });
      } catch (err) {
        // Convert technical errors to friendly messages
        let msg = err.message || 'Something went wrong.';
        if (msg.includes('401') || msg.toLowerCase().includes('unauthorized') || msg.toLowerCase().includes('api key')) {
          msg = `🔑 API key invalid or expired for ${providerLabel(activeProvider)}. Please re-enter it via the model selector.`;
        } else if (msg.includes('timed out') || msg.includes('timeout')) {
          msg = '⏱ The request timed out. Try a shorter question or check your connection.';
        } else if (msg.includes('502') || msg.toLowerCase().includes('could not reach')) {
          msg = `📡 Could not reach ${providerLabel(activeProvider)}. Check your internet or that Ollama is running.`;
        }

        if (streamedText.trim()) {
          resolveThinkingBubble({
            question: q,
            answer: `${streamedText}\n\nResponse interrupted`,
            sources: [],
            model: currentModel || defaultModel,
            model_response_ms: 0,
          });
        } else {
          failThinkingBubble(msg);
        }
      } finally {
        chatPending = false;
        askBtn.disabled = false;
        askBtn.textContent = 'Ask';
        question.classList.remove('loading');
      }
    });

    // ── Clear history ─────────────────────────────────
    clearHistoryBtn.addEventListener('click', async () => {
      if (!confirm('Delete all chat history?')) return;
      try {
        await fetchJson('/chat/history', { method: 'DELETE' });
        currentChatId = null;
        cachedSessions = [];
        attachmentsByChat.clear();
        draftAttachments = [];
        renderSessions([]);
        renderAttachments();
        mainTitle.textContent = 'New Chat';
        mainMeta.textContent = '';
        chatThread.innerHTML = `<div class="chat-empty">
          <div class="empty-illustration">✦</div>
          <div class="chat-empty-title">History cleared</div>
          <div class="chat-empty-hint">Start a fresh conversation any time with a new question.</div>
        </div>`;
      } catch (err) {
        alert('Failed to clear: ' + err.message);
      }
    });

    // ── Refresh ───────────────────────────────────────
    refreshBtn.addEventListener('click', () => loadHistory());

    // ── Init ──────────────────────────────────────────
    if ('requestIdleCallback' in window) {
      loadModelOptions();
      renderAttachments();
      updateComposerMeta();
      window.requestIdleCallback(() => loadHistory({ background: true }));
    } else {
      loadModelOptions();
      renderAttachments();
      updateComposerMeta();
      window.setTimeout(() => loadHistory({ background: true }), 0);
    }

