/**
 * VoxAI App Logic v2
 * Supports: Local, Cloud (RunPod), and Provider (Gemini) models
 * Features: Thinking sections, boot progress, web search, file uploads, multi-turn context
 */

// STATE
const AppState = {
    mode: 'chat', // 'chat' | 'image'

    // Model categories
    localModels: [],
    cloudModels: [],
    providerModels: [],

    // Active chat model
    currentModel: null,          // model ID (filename / hf_id / gemini name)
    currentModelDisplay: 'Chat', // display name
    currentModelSource: 'local', // 'local' | 'cloud' | 'provider'
    currentModelProvider: null,  // 'Gemini' etc.

    // Image models
    imageModels: [],
    currentImageModel: null,
    currentImageDisplay: 'No Model',

    currentTheme: 'ember',
    availableThemes: ['ember', 'arctic', 'violet', 'jade', 'sunset'],

    // Chat history (persisted in localStorage)
    chatHistory: [],
    activeChatId: null,

    // Conversation context (sent to server for multi-turn)
    conversationMessages: [],

    // File attachments pending send
    pendingFiles: [],

    // Web search mode
    searchMode: false,

    // Image gallery (sidebar thumbnails)
    galleryImages: []
};

// CONSTANTS
const CONSTANTS = {
    API: {
        CHAT: '/chat',
        CHAT_STREAM: '/chat/stream',
        GENERATE: '/api/image/generate',
        MODELS: '/api/models',
        LLM_MODELS: '/api/llm-models',
        CLOUD_TERMINATE: '/api/cloud/terminate',
        UPLOAD: '/api/upload',
        SEARCH: '/api/search'
    },
    THEMES_PATH: '/static/css/themes/',
    STORAGE_KEY_HISTORY: 'vox_chat_history',
    STORAGE_KEY_THEME: 'vox_theme',
    STORAGE_KEY_MODEL: 'vox_selected_model'
};

// IMAGE MODEL FAMILY PROFILES
const MODEL_PROFILES = {
    sd15:    { native: '512x512',  steps: 25, cfg: 7.0,  neg: true  },
    sd20:    { native: '768x768',  steps: 28, cfg: 7.5,  neg: true  },
    sdxl:    { native: '1024x1024', steps: 30, cfg: 5.5, neg: true  },
    flux:    { native: '1024x1024', steps: 20, cfg: 3.5, neg: false },
    pony:    { native: '1024x1024', steps: 25, cfg: 6.0, neg: true  },
    gemini:  { native: '1024x1024', steps: 1,  cfg: 1.0, neg: false },
    openai:  { native: '1024x1024', steps: 1,  cfg: 1.0, neg: false },
    video:   { native: '1280x720',  steps: 1,  cfg: 1.0, neg: false },
    unknown: { native: '512x512', steps: 25, cfg: 7.0, neg: true  }
};

// Known model stats for common models (approximate values)
const MODEL_STATS_MAP = [
    // Local GGUF patterns
    { re: /0\.5[bB].*Q4/i,   stats: { params: '0.5B', vram: '~0.5 GB', speed: '~120 t/s' }},
    { re: /1[bB][\-_\.].*Q4/i, stats: { params: '1B', vram: '~1.0 GB', speed: '~80 t/s' }},
    { re: /3[bB][\-_\.].*Q4/i, stats: { params: '3B', vram: '~2.5 GB', speed: '~45 t/s' }},
    { re: /[78][bB][\-_\.].*Q4/i, stats: { params: '7-8B', vram: '~5.5 GB', speed: '~25 t/s' }},
    { re: /[78][bB][\-_\.].*Q8/i, stats: { params: '7-8B', vram: '~9 GB', speed: '~15 t/s' }},
    { re: /1[234][bB][\-_\.].*Q4/i, stats: { params: '12-14B', vram: '~9 GB', speed: '~15 t/s' }},
    { re: /3[012][bB][\-_\.].*Q4/i, stats: { params: '27-32B', vram: '~20 GB', speed: '~9 t/s' }},
    { re: /7[012][bB][\-_\.].*Q4/i, stats: { params: '70-72B', vram: '~42 GB', speed: '~5 t/s' }},
    // Cloud/HF model patterns
    { re: /qwen.*0\.5b/i,  stats: { params: '0.5B', vram: '~1 GB', speed: '~200 t/s' }},
    { re: /qwen.*1\.5b/i,  stats: { params: '1.5B', vram: '~3 GB', speed: '~150 t/s' }},
    { re: /qwen.*3b/i,     stats: { params: '3B', vram: '~6 GB', speed: '~120 t/s' }},
    { re: /qwen.*7b/i,     stats: { params: '7B', vram: '~14 GB', speed: '~80 t/s' }},
    { re: /qwen.*14b/i,    stats: { params: '14B', vram: '~28 GB', speed: '~45 t/s' }},
    { re: /qwen.*32b/i,    stats: { params: '32B', vram: '~38 GB', speed: '~25 t/s' }},
    { re: /qwen.*72b/i,    stats: { params: '72B', vram: '~48 GB', speed: '~15 t/s' }},
    { re: /llama.*3b/i,    stats: { params: '3B', vram: '~6 GB', speed: '~120 t/s' }},
    { re: /llama.*8b/i,    stats: { params: '8B', vram: '~16 GB', speed: '~70 t/s' }},
    { re: /llama.*70b/i,   stats: { params: '70B', vram: '~48 GB', speed: '~15 t/s' }},
    { re: /mistral.*7b/i,  stats: { params: '7B', vram: '~14 GB', speed: '~80 t/s' }},
    { re: /mixtral.*8x7/i, stats: { params: '46.7B', vram: '~48 GB', speed: '~25 t/s' }},
    { re: /gemma.*2b/i,    stats: { params: '2B', vram: '~4 GB', speed: '~140 t/s' }},
    { re: /gemma.*9b/i,    stats: { params: '9B', vram: '~18 GB', speed: '~60 t/s' }},
    { re: /gemma.*27b/i,   stats: { params: '27B', vram: '~36 GB', speed: '~20 t/s' }},
    { re: /deepseek.*7b/i, stats: { params: '7B', vram: '~14 GB', speed: '~80 t/s' }},
    { re: /phi.*mini/i,    stats: { params: '3.8B', vram: '~8 GB', speed: '~100 t/s' }},
];

function _lookupModelStats(name) {
    if (!name) return {};
    for (const entry of MODEL_STATS_MAP) {
        if (entry.re.test(name)) return entry.stats;
    }
    return {};
}

// ========================================================================
// INIT
// ========================================================================
window.onload = function () {
    loadChatHistory();
    initTheme();
    setupEventListeners();
    fetchAllModels();
    restoreModel();
    autoResize(document.getElementById('user-input'));
    updateUI();
    renderHistoryList();
};

function setupEventListeners() {
    document.getElementById('nav-chat').onclick = () => switchMode('chat');
    document.getElementById('nav-image')?.addEventListener('click', () => switchMode('image'));
    document.getElementById('new-chat-btn').onclick = newChat;

    const input = document.getElementById('user-input');
    input.oninput = () => autoResize(input);
    input.onkeydown = (e) => handleKey(e);
    document.getElementById('send-btn').onclick = handleSubmit;

    const attachBtn = document.getElementById('attach-btn');
    const fileInput = document.getElementById('file-input');
    if (attachBtn && fileInput) {
        attachBtn.onclick = () => fileInput.click();
        fileInput.onchange = handleFileSelect;
    }

    const searchToggle = document.getElementById('search-toggle');
    if (searchToggle) searchToggle.onclick = toggleSearchMode;

    document.getElementById('model-selector').onclick = openModelModal;
    document.getElementById('settings-btn').onclick = toggleSettings;

    const modelModal = document.getElementById('model-modal') || document.getElementById('modal-overlay');
    if (modelModal) {
        modelModal.onclick = (e) => { if (e.target === modelModal) closeModelModal(); };
    }

    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) themeSelect.onchange = (e) => setTheme(e.target.value);
}

// ========================================================================
// THEME ENGINE
// ========================================================================
function initTheme() {
    const saved = localStorage.getItem(CONSTANTS.STORAGE_KEY_THEME);
    if (saved && AppState.availableThemes.includes(saved)) setTheme(saved);
    else setTheme('ember');
}

function setTheme(themeName) {
    if (!AppState.availableThemes.includes(themeName)) return;
    AppState.currentTheme = themeName;
    localStorage.setItem(CONSTANTS.STORAGE_KEY_THEME, themeName);
    let link = document.getElementById('theme-stylesheet');
    const newHref = `${CONSTANTS.THEMES_PATH}${themeName}.css`;
    if (!link) { link = document.createElement('link'); link.id = 'theme-stylesheet'; link.rel = 'stylesheet'; document.head.appendChild(link); }
    link.href = newHref;
    const dropdown = document.getElementById('theme-select');
    if (dropdown) dropdown.value = themeName;
}

// ========================================================================
// MODE SWITCHING
// ========================================================================
function switchMode(newMode) {
    AppState.mode = newMode;
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`nav-${newMode}`).classList.add('active');

    const input = document.getElementById('user-input');
    input.placeholder = newMode === 'chat' ? "Message VoxAI..." : "Describe the image you want to generate...";

    document.getElementById('chat-feed').innerHTML = '';
    addSystemMessage(`Switched to ${newMode.toUpperCase()} mode.`);

    const settingsPanel = document.getElementById('gen-settings');
    if (settingsPanel) settingsPanel.style.display = newMode === 'image' ? 'block' : 'none';

    // Toggle sidebar: chat history vs image gallery
    const histLabel = document.getElementById('sidebar-history-label');
    const histList = document.getElementById('chat-history-list');
    const galLabel = document.getElementById('sidebar-gallery-label');
    const gallery = document.getElementById('image-gallery');
    const newChatBtn = document.getElementById('new-chat-btn');

    if (newMode === 'image') {
        if (histLabel) histLabel.style.display = 'none';
        if (histList) histList.style.display = 'none';
        if (galLabel) galLabel.style.display = 'block';
        if (gallery) gallery.style.display = 'flex';
        if (newChatBtn) newChatBtn.style.display = 'none';
    } else {
        if (histLabel) histLabel.style.display = 'block';
        if (histList) histList.style.display = 'flex';
        if (galLabel) galLabel.style.display = 'none';
        if (gallery) gallery.style.display = 'none';
        if (newChatBtn) newChatBtn.style.display = 'block';
    }

    updateUI();
    input.focus();
}

// ========================================================================
// CHAT HISTORY
// ========================================================================
function generateChatId() { return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6); }

function loadChatHistory() {
    try { AppState.chatHistory = JSON.parse(localStorage.getItem(CONSTANTS.STORAGE_KEY_HISTORY) || '[]'); }
    catch (e) { AppState.chatHistory = []; }
}

function saveChatHistory() {
    try { if (AppState.chatHistory.length > 50) AppState.chatHistory = AppState.chatHistory.slice(0, 50); localStorage.setItem(CONSTANTS.STORAGE_KEY_HISTORY, JSON.stringify(AppState.chatHistory)); }
    catch (e) {}
}

function saveCurrentChat() {
    const feed = document.getElementById('chat-feed');
    const msgElements = feed.querySelectorAll('.message');
    if (msgElements.length === 0) return;

    const messages = [];
    msgElements.forEach(el => {
        const isUser = el.classList.contains('user-msg');
        const isAi = el.classList.contains('ai-msg');
        const content = el.querySelector('.msg-content');
        if (content && (isUser || isAi)) {
            messages.push({ role: isUser ? 'user' : 'ai', html: content.innerHTML });
        }
    });
    if (messages.length === 0) return;

    const firstUser = messages.find(m => m.role === 'user');
    const titleEl = document.createElement('div');
    titleEl.innerHTML = firstUser ? firstUser.html : 'Chat';
    let title = titleEl.textContent.trim().substring(0, 40) || 'Chat';

    if (AppState.activeChatId) {
        const idx = AppState.chatHistory.findIndex(c => c.id === AppState.activeChatId);
        if (idx !== -1) {
            AppState.chatHistory[idx].messages = messages;
            AppState.chatHistory[idx].title = title;
            AppState.chatHistory[idx].timestamp = Date.now();
            AppState.chatHistory[idx].context = AppState.conversationMessages;
        }
    } else {
        const chatObj = { id: generateChatId(), title, messages, context: AppState.conversationMessages, timestamp: Date.now() };
        AppState.chatHistory.unshift(chatObj);
        AppState.activeChatId = chatObj.id;
    }
    saveChatHistory();
    renderHistoryList();
}

function loadChat(chatId) {
    const feed = document.getElementById('chat-feed');
    if (feed.querySelectorAll('.message').length > 0 && AppState.activeChatId !== chatId) saveCurrentChat();

    const chat = AppState.chatHistory.find(c => c.id === chatId);
    if (!chat) return;

    feed.innerHTML = '';
    clearStats();
    AppState.activeChatId = chatId;
    AppState.conversationMessages = chat.context || [];

    chat.messages.forEach(m => {
        const div = document.createElement('div');
        div.className = `message ${m.role}-msg`;
        div.innerHTML = `<div class="avatar">${m.role === 'user' ? 'You' : 'V'}</div><div class="msg-content">${m.html}</div>`;
        feed.appendChild(div);
    });
    scrollToBottom();
    renderHistoryList();
}

function deleteChat(chatId, event) {
    event.stopPropagation();
    AppState.chatHistory = AppState.chatHistory.filter(c => c.id !== chatId);
    saveChatHistory();
    if (AppState.activeChatId === chatId) {
        AppState.activeChatId = null;
        AppState.conversationMessages = [];
        document.getElementById('chat-feed').innerHTML = '';
        clearStats();
    }
    renderHistoryList();
}

function renderHistoryList() {
    const container = document.getElementById('chat-history-list');
    if (!container) return;
    container.innerHTML = '';
    AppState.chatHistory.forEach(chat => {
        const div = document.createElement('div');
        div.className = `chat-history-item ${chat.id === AppState.activeChatId ? 'active' : ''}`;
        div.innerHTML = `<span style="overflow:hidden;text-overflow:ellipsis;">${chat.title}</span><span class="delete-chat" onclick="deleteChat('${chat.id}', event)">✕</span>`;
        div.onclick = () => loadChat(chat.id);
        container.appendChild(div);
    });
}

function newChat() {
    const feed = document.getElementById('chat-feed');
    if (feed.querySelectorAll('.message').length > 0) saveCurrentChat();
    feed.innerHTML = '';
    clearStats();
    AppState.activeChatId = null;
    AppState.conversationMessages = [];
    renderHistoryList();
    document.getElementById('user-input').focus();
}

// ========================================================================
// API & MODEL MANAGEMENT
// ========================================================================
async function fetchAllModels() {
    const [llmResult, imgResult] = await Promise.allSettled([
        fetch(CONSTANTS.API.LLM_MODELS).then(r => r.json()),
        fetch(CONSTANTS.API.MODELS).then(r => r.json())
    ]);

    if (llmResult.status === 'fulfilled') {
        const data = llmResult.value;
        AppState.localModels = data.models || [];
        AppState.cloudModels = data.cloud_models || [];
        AppState.providerModels = data.provider_models || [];

        // Auto-select first available if none selected
        if (!AppState.currentModel) {
            if (AppState.localModels.length > 0) selectChatModel(AppState.localModels[0].name, AppState.localModels[0].display, 'local');
            else if (AppState.cloudModels.length > 0) selectChatModel(AppState.cloudModels[0].name, AppState.cloudModels[0].display, 'cloud');
            else if (AppState.providerModels.length > 0) selectChatModel(AppState.providerModels[0].name, AppState.providerModels[0].display, 'provider', AppState.providerModels[0].provider);
        }
    }

    if (imgResult.status === 'fulfilled') {
        AppState.imageModels = imgResult.value.models || [];
        if (AppState.imageModels.length > 0) {
            AppState.currentImageModel = imgResult.value.current || AppState.imageModels[0].name;
            AppState.currentImageDisplay = imgResult.value.current_display || AppState.imageModels[0].display;
        }
    }
    updateUI();
}

function restoreModel() {
    try {
        const saved = localStorage.getItem(CONSTANTS.STORAGE_KEY_MODEL);
        if (saved) { const m = JSON.parse(saved); AppState.currentModel = m.name; AppState.currentModelDisplay = m.display; AppState.currentModelSource = m.source; AppState.currentModelProvider = m.provider || null; }
    } catch (e) {}
}

function selectChatModel(name, display, source, provider) {
    const prevSource = AppState.currentModelSource;
    const prevModel = AppState.currentModel;

    AppState.currentModel = name;
    AppState.currentModelDisplay = display;
    AppState.currentModelSource = source || 'local';
    AppState.currentModelProvider = provider || null;
    localStorage.setItem(CONSTANTS.STORAGE_KEY_MODEL, JSON.stringify({ name, display, source: AppState.currentModelSource, provider: AppState.currentModelProvider }));
    updateUI();

    // Terminate RunPod pod when switching away from cloud or to a different cloud model
    if (prevSource === 'cloud' && (source !== 'cloud' || prevModel !== name)) {
        fetch(CONSTANTS.API.CLOUD_TERMINATE, { method: 'POST' }).catch(() => {});
    }
}

function getCurrentDisplay() {
    if (AppState.mode === 'chat') {
        const badge = AppState.currentModelSource === 'cloud' ? ' ☁️' : AppState.currentModelSource === 'provider' ? ' ⚡' : '';
        return (AppState.currentModelDisplay || 'Chat') + badge;
    }
    return AppState.currentImageDisplay;
}

// ========================================================================
// SUBMIT
// ========================================================================
async function handleSubmit() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text && AppState.pendingFiles.length === 0) return;

    const attachHtml = buildAttachmentPreviewHtml();
    addMessage('user', text, attachHtml);
    input.value = '';
    input.style.height = 'auto';

    if (AppState.mode === 'chat') {
        let uploadedFiles = [];
        if (AppState.pendingFiles.length > 0) { uploadedFiles = await uploadFiles(); clearAttachments(); }
        await handleChat(text, uploadedFiles);
        saveCurrentChat();
    } else {
        clearAttachments();
        await handleImage(text);
    }
}

function buildAttachmentPreviewHtml() {
    if (AppState.pendingFiles.length === 0) return '';
    let html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">';
    AppState.pendingFiles.forEach(entry => {
        if (entry.type === 'image' && entry.preview) html += `<img src="${entry.preview}" style="max-width:120px;max-height:80px;border-radius:6px;border:1px solid var(--border);" alt="${entry.file.name}">`;
        else html += `<span style="background:rgba(255,255,255,0.06);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:11px;">📄 ${entry.file.name}</span>`;
    });
    return html + '</div>';
}

// ========================================================================
// CHAT STREAMING (core feature — thinking, boot progress, multi-turn)
// ========================================================================
async function handleChat(text, uploadedFiles) {
    const typingId = addTyping();
    clearStats();

    // Build message with file context
    let fullMessage = text;
    if (uploadedFiles && uploadedFiles.length > 0) {
        const fileContext = uploadedFiles.map(f => {
            if (f.type === 'text') return `[File: ${f.filename}]\n\`\`\`\n${f.content}\n\`\`\``;
            else if (f.type === 'image') return `[Image attached: ${f.filename}]`;
            return `[Attached: ${f.filename}]`;
        }).join('\n\n');
        fullMessage = fileContext + (text ? '\n\n' + text : '\n\nPlease analyze the attached files.');
    }

    // Web search check
    const searchQuery = AppState.searchMode ? text : detectSearchIntent(text);
    if (searchQuery) {
        if (AppState.searchMode) {
            AppState.searchMode = false;
            const btn = document.getElementById('search-toggle');
            if (btn) btn.classList.remove('active');
            document.getElementById('user-input').placeholder = 'Message VoxAI...';
        }
        const results = await webSearch(searchQuery);
        if (results.length > 0) { addSearchResultCard(results, searchQuery); fullMessage = text + formatSearchResults(results); }
    }

    // Add to conversation context
    AppState.conversationMessages.push({ role: 'user', content: fullMessage });

    try {
        const resp = await fetch(CONSTANTS.API.CHAT_STREAM, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: fullMessage,
                model: AppState.currentModel,
                source: AppState.currentModelSource,
                provider: AppState.currentModelProvider || '',
                history: AppState.conversationMessages.slice(0, -1),
                system_prompt: null
            })
        });

        if (!resp.ok) {
            removeElement(typingId);
            const fallback = await fetch(CONSTANTS.API.CHAT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: fullMessage, model: AppState.currentModel })
            });
            const data = await fallback.json();
            const reply = data.reply || data.error || "No response.";
            addMessage('ai', reply);
            AppState.conversationMessages.push({ role: 'assistant', content: reply });
            return;
        }

        // --- SSE Streaming ---
        removeElement(typingId);

        const feed = document.getElementById('chat-feed');
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-msg';
        msgDiv.innerHTML = `<div class="avatar">V</div><div class="msg-content"><p>...</p></div>`;
        feed.appendChild(msgDiv);
        scrollToBottom();

        const contentEl = msgDiv.querySelector('.msg-content');
        let fullText = '';
        let thinkText = '';
        let isThinking = false;
        let thinkSection = null;
        let bootIndicator = null;
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.slice(6).trim();
                if (!dataStr) continue;

                try {
                    const ev = JSON.parse(dataStr);

                    // --- BOOT PROGRESS (Countdown Timer) ---
                    if (ev.type === 'boot_start') {
                        bootIndicator = document.createElement('div');
                        bootIndicator.className = 'boot-indicator';
                        bootIndicator.dataset.eta = ev.eta || 235; // estimated total seconds
                        const eta = parseInt(bootIndicator.dataset.eta);
                        const em = Math.floor(eta / 60), es = eta % 60;
                        bootIndicator.innerHTML = `<span class="boot-spinner"></span> Booting ${ev.text}...`
                            + `<span class="boot-countdown">~${em}m ${es.toString().padStart(2,'0')}s remaining</span>`;
                        contentEl.innerHTML = '';
                        contentEl.appendChild(bootIndicator);
                        scrollToBottom();
                        continue;
                    }
                    if (ev.type === 'boot_tick') {
                        if (bootIndicator) {
                            const eta = parseInt(bootIndicator.dataset.eta) || 235;
                            let remaining = Math.max(0, eta - ev.elapsed);
                            const rm = Math.floor(remaining / 60), rs = remaining % 60;
                            const label = remaining > 0
                                ? `~${rm}m ${rs.toString().padStart(2,'0')}s remaining`
                                : 'Almost ready...';
                            bootIndicator.innerHTML = `<span class="boot-spinner"></span> Booting cloud pod...`
                                + `<span class="boot-countdown">${label}</span>`;
                        }
                        continue;
                    }
                    if (ev.type === 'boot_done') {
                        if (bootIndicator) { bootIndicator.remove(); bootIndicator = null; }
                        contentEl.innerHTML = '<p>...</p>';
                        continue;
                    }
                    if (ev.type === 'boot_fail') {
                        if (bootIndicator) bootIndicator.remove();
                        contentEl.innerHTML = '<p style="color:#f87171;">Cloud boot failed. Check RunPod balance and API key.</p>';
                        AppState.conversationMessages.push({ role: 'assistant', content: 'Cloud boot failed.' });
                        return;
                    }

                    // --- THINKING SECTIONS ---
                    if (ev.type === 'think_start') {
                        isThinking = true;
                        thinkText = '';
                        thinkSection = document.createElement('details');
                        thinkSection.className = 'thinking-section';
                        thinkSection.setAttribute('open', '');
                        thinkSection.innerHTML = `<summary><span class="think-icon">💭</span> Thinking...</summary><div class="think-content"></div>`;
                        contentEl.insertBefore(thinkSection, contentEl.firstChild);
                        scrollToBottom();
                        continue;
                    }
                    if (ev.type === 'think_chunk') {
                        thinkText += ev.text;
                        if (thinkSection) {
                            const tc = thinkSection.querySelector('.think-content');
                            if (tc) tc.innerHTML = typeof marked !== 'undefined' ? marked.parse(thinkText) : `<p>${thinkText}</p>`;
                        }
                        scrollToBottom();
                        continue;
                    }
                    if (ev.type === 'think_end') {
                        isThinking = false;
                        if (thinkSection) {
                            thinkSection.removeAttribute('open');
                            const sum = thinkSection.querySelector('summary');
                            if (sum) sum.innerHTML = `<span class="think-icon">💭</span> Thought for a moment`;
                        }
                        // Clear placeholder
                        const ph = contentEl.querySelector(':scope > p');
                        if (ph && ph.textContent === '...') ph.remove();
                        continue;
                    }

                    // --- CONTENT ---
                    if (ev.type === 'chunk') {
                        fullText += ev.text;
                        const rendered = typeof marked !== 'undefined' ? marked.parse(fullText) : `<p>${fullText}</p>`;
                        if (thinkSection) {
                            // Keep thinking section at top, add rendered text below
                            const textContainer = contentEl.querySelector('.response-text') || document.createElement('div');
                            textContainer.className = 'response-text';
                            textContainer.innerHTML = rendered;
                            if (!contentEl.querySelector('.response-text')) contentEl.appendChild(textContainer);
                        } else {
                            contentEl.innerHTML = rendered;
                        }
                        scrollToBottom();
                    } else if (ev.type === 'done') {
                        showStats(ev.stats);
                    } else if (ev.type === 'error') {
                        fullText += ev.text;
                        contentEl.innerHTML = `<p style="color:#f87171;">${ev.text}</p>`;
                    }
                } catch (e) { /* JSON parse error — skip */ }
            }
        }

        // Process remaining buffer
        if (buffer.startsWith('data: ')) {
            try { const ev = JSON.parse(buffer.slice(6).trim()); if (ev.type === 'done') showStats(ev.stats); } catch (e) {}
        }

        // Add downloadable file cards for code blocks
        attachCodeFileCards(contentEl, fullText);

        // Save to conversation context
        AppState.conversationMessages.push({ role: 'assistant', content: fullText });
        // Limit context
        if (AppState.conversationMessages.length > 20) AppState.conversationMessages = AppState.conversationMessages.slice(-20);

    } catch (e) {
        removeElement(typingId);
        addMessage('ai', "[Error] Could not connect to AI Backend.");
    }
}

// ========================================================================
// IMAGE GENERATION
// ========================================================================
async function handleImage(prompt) {
    const loadingId = addTyping();
    clearStats();
    const startTime = performance.now();

    const modelName = (AppState.currentImageModel || '').toLowerCase();
    const isVideo = modelName.startsWith('sora-') || modelName.startsWith('veo-');

    if (isVideo) {
        // --- Video Generation ---
        const payload = {
            prompt,
            model: AppState.currentImageModel,
            duration: 8,
            aspect: '16:9',
        };
        try {
            const resp = await fetch('/api/video/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            removeElement(loadingId);
            if (data.status === 'success') {
                const duration = ((performance.now() - startTime) / 1000).toFixed(1);
                addVideoCard(data.url, prompt, data.filename);
                addToGallery(data.url, prompt, data.filename);
                showImageStats({ duration, resolution: 'Video', steps: '-', model: AppState.currentImageDisplay });
            } else {
                addMessage('ai', `[Video Error] ${data.error}`);
            }
        } catch (e) { removeElement(loadingId); addMessage('ai', `[System Error] ${e.message}`); }
    } else {
        // --- Image Generation ---
        const presetVal = document.getElementById('img-preset')?.value || '1024x1024';
        const [w, h] = presetVal.split('x').map(Number);
        const steps = parseInt(document.getElementById('img-steps')?.value || 20);
        const payload = {
            prompt, negative_prompt: document.getElementById('img-neg')?.value || '',
            width: w, height: h, steps: steps,
            cfg: document.getElementById('img-cfg')?.value || 7.0, model: AppState.currentImageModel
        };
        try {
            const resp = await fetch(CONSTANTS.API.GENERATE, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await resp.json();
            removeElement(loadingId);
            if (data.status === 'success') {
                const duration = ((performance.now() - startTime) / 1000).toFixed(1);
                addImageCard(data.url, prompt, data.filename);
                addToGallery(data.url, prompt, data.filename);
                showImageStats({ duration, resolution: `${w}x${h}`, steps, model: AppState.currentImageDisplay });
            }
            else addMessage('ai', `[Image Error] ${data.error}`);
        } catch (e) { removeElement(loadingId); addMessage('ai', `[System Error] ${e.message}`); }
    }
}

// ========================================================================
// STATS
// ========================================================================
function showStats(stats) {
    const el = document.getElementById('chat-stats');
    if (!el || !stats) return;
    el.textContent = `${stats.speed} t/s  ·  ${stats.tokens} tokens  ·  ${stats.duration}s`;
    el.style.display = 'block';
}

function clearStats() {
    const el = document.getElementById('chat-stats');
    if (el) { el.textContent = ''; el.style.display = 'none'; }
}

function showImageStats(info) {
    const el = document.getElementById('chat-stats');
    if (!el || !info) return;
    el.textContent = `${info.resolution}  ·  ${info.steps} steps  ·  ${info.duration}s  ·  ${info.model}`;
    el.style.display = 'block';
}

// ========================================================================
// IMAGE GALLERY (sidebar)
// ========================================================================
function addToGallery(url, prompt, filename) {
    AppState.galleryImages.unshift({ url, prompt, filename });
    // Limit gallery to 50 images
    if (AppState.galleryImages.length > 50) AppState.galleryImages.pop();
    renderGallery();
}

function renderGallery() {
    const grid = document.getElementById('gallery-grid');
    const empty = document.getElementById('gallery-empty');
    if (!grid) return;

    if (AppState.galleryImages.length === 0) {
        grid.innerHTML = '';
        if (empty) empty.style.display = 'block';
        return;
    }

    if (empty) empty.style.display = 'none';
    grid.innerHTML = '';

    AppState.galleryImages.forEach(img => {
        const thumb = document.createElement('div');
        thumb.className = 'gallery-thumb';
        const shortPrompt = img.prompt.length > 30 ? img.prompt.substring(0, 30) + '...' : img.prompt;
        const isVid = img.filename && img.filename.endsWith('.mp4');
        if (isVid) {
            thumb.innerHTML = `<div style="position:relative"><video src="${img.url}" muted preload="metadata" style="width:100%;border-radius:4px;"></video><div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:24px;color:white;text-shadow:0 0 6px rgba(0,0,0,0.8);">▶</div></div><div class="gallery-label">${shortPrompt}</div>`;
        } else {
            thumb.innerHTML = `<img src="${img.url}" alt="${shortPrompt}" loading="lazy"><div class="gallery-label">${shortPrompt}</div>`;
        }
        thumb.onclick = () => window.open(img.url, '_blank');
        grid.appendChild(thumb);
    });
}

// ========================================================================
// DOM HELPERS
// ========================================================================
function updateUI() {
    const el = document.getElementById('current-model-name');
    if (el) el.innerText = getCurrentDisplay();
}

function addMessage(role, text, extraHtml) {
    const feed = document.getElementById('chat-feed');
    const div = document.createElement('div');
    div.className = `message ${role}-msg`;
    const avatarTxt = role === 'user' ? 'You' : 'V';
    const inner = role === 'ai' && typeof marked !== 'undefined' ? marked.parse(text) : `<p>${text}</p>`;
    div.innerHTML = `<div class="avatar">${avatarTxt}</div><div class="msg-content">${inner}${extraHtml || ''}</div>`;
    feed.appendChild(div);
    scrollToBottom();
}

function addImageCard(url, prompt, filename) {
    const feed = document.getElementById('chat-feed');
    const div = document.createElement('div');
    div.className = 'message ai-msg';
    div.innerHTML = `
        <div class="avatar">IMG</div>
        <div class="msg-content" style="width:100%">
            <p>Generated Image:</p>
            <div class="image-card">
                <img src="${url}" onclick="window.open('${url}', '_blank')">
                <div class="image-meta">
                    <span>${prompt.substring(0, 40)}${prompt.length > 40 ? '...' : ''}</span>
                    <a href="${url}" download="${filename}" class="download-link">Download</a>
                </div>
            </div>
        </div>`;
    feed.appendChild(div);
    scrollToBottom();
}

function addVideoCard(url, prompt, filename) {
    const feed = document.getElementById('chat-feed');
    const div = document.createElement('div');
    div.className = 'message ai-msg';
    div.innerHTML = `
        <div class="avatar">VID</div>
        <div class="msg-content" style="width:100%">
            <p>Generated Video:</p>
            <div class="video-card">
                <video controls preload="metadata" style="width:100%;max-height:420px;border-radius:8px;background:black;">
                    <source src="${url}" type="video/mp4">
                    Your browser does not support video playback.
                </video>
                <div class="image-meta" style="margin-top:6px;">
                    <span>${prompt.substring(0, 40)}${prompt.length > 40 ? '...' : ''}</span>
                    <a href="${url}" download="${filename}" class="download-link">Download</a>
                </div>
            </div>
        </div>`;
    feed.appendChild(div);
    scrollToBottom();
}

function addTyping() {
    const id = 'typing-' + Date.now();
    document.getElementById('chat-feed').insertAdjacentHTML('beforeend', `
        <div id="${id}" class="message ai-msg typing-anim" style="opacity:0.6;">
            <div class="avatar">...</div>
            <div class="msg-content"><p>Thinking...</p></div>
        </div>`);
    scrollToBottom();
    return id;
}

function addSystemMessage(text) {
    document.getElementById('chat-feed').insertAdjacentHTML('beforeend',
        `<div style="text-align:center; opacity:0.5; font-size:12px; margin:20px 0;">${text}</div>`);
}

function removeElement(id) { const el = document.getElementById(id); if (el) el.remove(); }
function scrollToBottom() { const a = document.getElementById('scroll-area'); if (a) a.scrollTop = a.scrollHeight; }
function autoResize(el) { if (!el) return; el.style.height = 'auto'; el.style.height = el.scrollHeight + 'px'; }
function handleKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); } }

function toggleSettings() {
    const p = document.getElementById('gen-settings');
    if (p) p.style.display = (p.style.display === 'none' || p.style.display === '') ? 'block' : 'none';
}

// ========================================================================
// MODEL MODAL (3 Tabs: Local / Cloud / Providers)
// ========================================================================
function openModelModal() {
    const modal = document.getElementById('model-modal') || document.getElementById('modal-overlay');
    const list = document.getElementById('model-list-container');
    const title = document.getElementById('model-modal-title');
    const tabsBar = document.getElementById('model-tabs');
    if (!modal || !list) return;
    modal.style.display = 'flex';

    if (AppState.mode === 'image') {
        if (title) title.textContent = 'Select Image Model';
        if (tabsBar) tabsBar.style.display = 'none';
        renderImageModelList(list);
        return;
    }

    if (title) title.textContent = 'Select LLM Model';
    if (tabsBar) tabsBar.style.display = 'flex';

    // Update tab labels with counts
    tabsBar.querySelectorAll('.model-tab-btn').forEach(btn => {
        const tab = btn.getAttribute('data-tab');
        if (tab === 'local') btn.textContent = `Local (${AppState.localModels.length})`;
        else if (tab === 'cloud') btn.textContent = `☁️ Cloud (${AppState.cloudModels.length})`;
        else if (tab === 'provider') btn.textContent = `⚡ Providers (${AppState.providerModels.length})`;
    });

    // Activate the tab matching current source
    switchModelTab(AppState.currentModelSource);
}

function switchModelTab(tabId) {
    const tabsBar = document.getElementById('model-tabs');
    const list = document.getElementById('model-list-container');
    if (!tabsBar || !list) return;

    tabsBar.querySelectorAll('.model-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    renderModelTab(list, tabId);
}

function renderModelTab(container, source) {
    container.innerHTML = '';
    let models = source === 'local' ? AppState.localModels : source === 'cloud' ? AppState.cloudModels : AppState.providerModels;

    if (models.length === 0) {
        const msg = source === 'cloud' ? 'No cloud models configured. Add in desktop Settings.'
            : source === 'provider' ? 'No provider API key configured.' : 'No local .gguf models found.';
        container.innerHTML = `<div style="padding:20px;text-align:center;opacity:0.5;">${msg}</div>`;
        return;
    }

    models.forEach(m => {
        const isSel = m.name === AppState.currentModel && source === AppState.currentModelSource;
        const div = document.createElement('div');
        div.className = `model-option ${isSel ? 'selected' : ''}`;

        // Build stats line
        let statsParts = [];
        if (source === 'local' && m.size_gb) statsParts.push(`${m.size_gb.toFixed(1)} GB`);
        if (source === 'cloud') statsParts.push('RunPod GPU');
        if (source === 'provider') statsParts.push(m.provider || 'API');

        // Look up known model stats from name
        const knownStats = _lookupModelStats(m.name);
        if (knownStats.params) statsParts.push(knownStats.params);
        if (knownStats.vram) statsParts.push(`VRAM ${knownStats.vram}`);
        if (knownStats.speed) statsParts.push(knownStats.speed);

        const statsLine = statsParts.length ? `<div style="font-size:10px;color:#00cccc;margin-top:2px;">${statsParts.join('  ·  ')}</div>` : '';

        div.innerHTML = `<div style="width:100%;"><div style="display:flex;justify-content:space-between;align-items:center;"><span style="font-weight:${isSel ? '600' : '400'}">${m.display}</span></div>${statsLine}</div>`;
        div.onclick = () => { selectChatModel(m.name, m.display, source, m.provider); closeModelModal(); };
        container.appendChild(div);
    });
}

function renderImageModelList(container) {
    container.innerHTML = '';
    if (AppState.imageModels.length === 0) { container.innerHTML = '<div style="padding:20px;text-align:center;opacity:0.5;">No image models found</div>'; return; }
    AppState.imageModels.forEach(m => {
        const isSel = m.name === AppState.currentImageModel;
        const div = document.createElement('div');
        div.className = `model-option ${isSel ? 'selected' : ''}`;
        div.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;width:100%;"><span style="font-weight:${isSel ? '600' : '400'}">${m.display}</span><span style="font-size:11px;opacity:0.5;">${m.family || ''} · ${(m.size_gb||0).toFixed(1)} GB</span></div>`;
        div.onclick = () => { AppState.currentImageModel = m.name; AppState.currentImageDisplay = m.display; applyModelProfile(m.name, m.family); updateUI(); closeModelModal(); };
        container.appendChild(div);
    });
}

function closeModelModal() {
    const modal = document.getElementById('model-modal') || document.getElementById('modal-overlay');
    if (modal) modal.style.display = 'none';
}

function applyModelProfile(modelName, knownFamily) {
    let family = knownFamily;
    if (!family) {
        const l = (modelName || '').toLowerCase();
        if (l.startsWith('sora-') || l.startsWith('veo-')) family = 'video';
        else if (l.startsWith('gemini-')) family = 'gemini';
        else if (l.startsWith('dall-e') || l.startsWith('gpt-image')) family = 'openai';
        else if (l.includes('flux')) family = 'flux';
        else if (l.includes('sdxl') || l.includes('xl_') || l.includes('_xl')) family = 'sdxl';
        else if (l.includes('pony') || l.includes('illustrious') || l.includes('animagine')) family = 'pony';
        else if (l.includes('sd2') || l.includes('v2-') || l.includes('768')) family = 'sd20';
        else family = 'sd15';
    }
    const p = MODEL_PROFILES[family] || MODEL_PROFILES.unknown;
    const preset = document.getElementById('img-preset'); if (preset) preset.value = p.native;
    const steps = document.getElementById('img-steps'); if (steps) steps.value = p.steps;
    const cfg = document.getElementById('img-cfg'); if (cfg) cfg.value = p.cfg;
    const neg = document.getElementById('img-neg'); if (neg) { neg.parentElement.style.display = p.neg ? '' : 'none'; if (!p.neg) neg.value = ''; }
}

// ========================================================================
// FILE ATTACHMENTS
// ========================================================================
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    files.forEach(file => {
        const isImg = file.type.startsWith('image/');
        const entry = { file, preview: null, type: isImg ? 'image' : 'text' };
        if (isImg) {
            const reader = new FileReader();
            reader.onload = (ev) => { entry.preview = ev.target.result; AppState.pendingFiles.push(entry); renderAttachPreview(); };
            reader.readAsDataURL(file);
        } else { AppState.pendingFiles.push(entry); renderAttachPreview(); }
    });
    e.target.value = '';
}

function renderAttachPreview() {
    const c = document.getElementById('attach-preview');
    if (!c) return;
    if (AppState.pendingFiles.length === 0) { c.style.display = 'none'; c.innerHTML = ''; return; }
    c.style.display = 'flex'; c.innerHTML = '';
    AppState.pendingFiles.forEach((entry, idx) => {
        const chip = document.createElement('div'); chip.className = 'attach-chip';
        if (entry.type === 'image' && entry.preview) chip.innerHTML = `<img src="${entry.preview}" alt="preview"><span>${entry.file.name}</span><span class="remove-attach" onclick="removeAttachment(${idx})">✕</span>`;
        else { const ext = entry.file.name.split('.').pop().toUpperCase(); chip.innerHTML = `<span style="font-size:16px;">📄</span><span>${entry.file.name} <span style="opacity:0.4;font-size:10px;">${ext}</span></span><span class="remove-attach" onclick="removeAttachment(${idx})">✕</span>`; }
        c.appendChild(chip);
    });
}

function removeAttachment(idx) { AppState.pendingFiles.splice(idx, 1); renderAttachPreview(); }
function clearAttachments() { AppState.pendingFiles = []; renderAttachPreview(); }

async function uploadFiles() {
    if (AppState.pendingFiles.length === 0) return [];
    const fd = new FormData();
    AppState.pendingFiles.forEach(e => fd.append('files', e.file));
    try { const r = await fetch(CONSTANTS.API.UPLOAD, { method: 'POST', body: fd }); if (!r.ok) return []; return (await r.json()).files || []; }
    catch (e) { console.error('Upload failed:', e); return []; }
}

// ========================================================================
// WEB SEARCH
// ========================================================================
function toggleSearchMode() {
    AppState.searchMode = !AppState.searchMode;
    const btn = document.getElementById('search-toggle');
    const input = document.getElementById('user-input');
    if (btn) btn.classList.toggle('active', AppState.searchMode);
    if (input) input.placeholder = AppState.searchMode ? 'Search the web...' : 'Message VoxAI...';
}

async function webSearch(query) {
    try {
        const r = await fetch(CONSTANTS.API.SEARCH, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, max_results: 5 }) });
        if (!r.ok) { addMessage('ai', `[Search Error] Status ${r.status}`); return []; }
        return (await r.json()).results || [];
    } catch (e) { addMessage('ai', `[Search Error] ${e.message}`); return []; }
}

function detectSearchIntent(text) {
    const lower = text.toLowerCase().trim();
    if (lower.length < 10) return null;
    const pats = [/^search\s+(?:for\s+)?(.+)/i, /^look\s+up\s+(.+)/i, /^google\s+(.+)/i, /^find\s+(?:info|information)\s+(?:on|about)\s+(.+)/i];
    for (const p of pats) { const m = text.match(p); if (m) { const q = m[1].replace(/\?+$/, '').trim(); if (q.length >= 5) return q; } }
    const conv = [/(?:can|could|would|please)?\s*(?:you\s+)?search\s+(?:online|the\s+web)\s+(?:for\s+)(.+)/i, /(?:can|could|would|please)?\s*(?:you\s+)?search\s+for\s+(.+)/i, /(?:can|could|would|please)?\s*(?:you\s+)?look\s+up\s+(.+)/i];
    for (const p of conv) { const m = text.match(p); if (m) { const q = m[1].replace(/\?+$/, '').trim(); if (q.length >= 5) return q; } }
    if (lower.includes('search the web') || lower.includes('search online')) {
        const q = text.replace(/(?:can|could|would|please)?\s*(?:you\s+)?search\s+(?:online|the\s+web|the\s+internet)\s*(?:for|and)?\s*/i, '').replace(/\?+$/, '').trim();
        return q.length >= 5 ? q : text.replace(/\?+$/, '').trim();
    }
    return null;
}

function formatSearchResults(results) {
    if (!results.length) return '';
    let ctx = '\n\n---\n**Web Search Results:**\n';
    results.forEach((r, i) => { ctx += `\n${i+1}. **${r.title}**\n   ${r.snippet}\n   Source: ${r.url}\n`; });
    return ctx + '---\nPlease use the search results above to answer. Cite sources.';
}

function addSearchResultCard(results, query) {
    const feed = document.getElementById('chat-feed');
    const div = document.createElement('div');
    div.className = 'message ai-msg';
    const html = results.map(r => `<div style="margin:6px 0;padding:8px 10px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid var(--border);"><div style="font-size:13px;font-weight:500;color:var(--text-primary);">${r.title}</div><div style="font-size:12px;color:var(--text-secondary);margin-top:3px;">${r.snippet}</div><div style="font-size:11px;opacity:0.4;margin-top:3px;">${r.url}</div></div>`).join('');
    div.innerHTML = `<div class="avatar">🔍</div><div class="msg-content"><p style="font-size:12px;opacity:0.6;margin-bottom:6px;">Search results for: "${query}"</p>${html}</div>`;
    feed.appendChild(div);
    scrollToBottom();
}

// ========================================================================
// CODE FILE CARDS (Downloadable code blocks)
// ========================================================================
const LANG_ICONS = {
    'python': '🐍', 'py': '🐍', 'javascript': '📜', 'js': '📜',
    'typescript': '📘', 'ts': '📘', 'html': '🌐', 'css': '🎨',
    'json': '📋', 'yaml': '📋', 'yml': '📋', 'xml': '📋',
    'bash': '🖥️', 'sh': '🖥️', 'shell': '🖥️', 'bat': '🖥️',
    'sql': '🗄️', 'java': '☕', 'cpp': '⚙️', 'c': '⚙️',
    'rust': '🦀', 'rs': '🦀', 'go': '🐹', 'ruby': '💎', 'rb': '💎',
    'php': '🐘', 'swift': '🐦', 'kotlin': '🟣', 'lua': '🌙',
    'toml': '📋', 'ini': '📋', 'cfg': '📋', 'dockerfile': '🐳',
    'makefile': '🔧', 'markdown': '📝', 'md': '📝',
};

const LANG_EXTENSIONS = {
    'python': '.py', 'py': '.py', 'javascript': '.js', 'js': '.js',
    'typescript': '.ts', 'ts': '.ts', 'html': '.html', 'css': '.css',
    'json': '.json', 'yaml': '.yaml', 'yml': '.yml', 'xml': '.xml',
    'bash': '.sh', 'sh': '.sh', 'shell': '.sh', 'bat': '.bat',
    'sql': '.sql', 'java': '.java', 'cpp': '.cpp', 'c': '.c',
    'rust': '.rs', 'rs': '.rs', 'go': '.go', 'ruby': '.rb', 'rb': '.rb',
    'php': '.php', 'swift': '.swift', 'kotlin': '.kt', 'lua': '.lua',
    'toml': '.toml', 'ini': '.ini', 'cfg': '.cfg', 'dockerfile': 'Dockerfile',
    'makefile': 'Makefile', 'markdown': '.md', 'md': '.md',
};

function attachCodeFileCards(contentEl, rawText) {
    // Parse code blocks from the raw markdown text
    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
    let match;
    let blockIndex = 0;

    while ((match = codeBlockRegex.exec(rawText)) !== null) {
        const lang = (match[1] || 'text').toLowerCase();
        const code = match[2];
        if (code.trim().length < 10) continue; // Skip trivial snippets

        blockIndex++;
        const ext = LANG_EXTENSIONS[lang] || '.txt';
        const icon = LANG_ICONS[lang] || '📄';
        const langName = lang.charAt(0).toUpperCase() + lang.slice(1);
        const filename = `code_${blockIndex}${ext}`;
        const lines = code.split('\n').length;
        const sizeKb = (new Blob([code]).size / 1024).toFixed(1);

        const card = document.createElement('div');
        card.className = 'code-file-card';
        card.title = `Download ${filename}`;
        card.innerHTML = `
            <div class="code-file-icon">${icon}</div>
            <div class="code-file-info">
                <div class="code-file-name">${filename}</div>
                <div class="code-file-meta">${langName}  ·  ${lines} lines  ·  ${sizeKb} KB</div>
            </div>
            <button class="code-file-download" title="Download">⬇</button>
        `;

        // Click to download
        const downloadFn = () => {
            const blob = new Blob([code], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        };

        card.querySelector('.code-file-download').onclick = (e) => { e.stopPropagation(); downloadFn(); };
        card.onclick = downloadFn;

        contentEl.appendChild(card);
    }
}
