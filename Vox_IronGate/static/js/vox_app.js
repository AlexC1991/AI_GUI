/**
 * VoxAI App Logic
 * Modularized for easier maintenance.
 */

// STATE
const AppState = {
    mode: 'chat', // 'chat' | 'image'

    // Chat (LLM) models
    chatModels: [],
    currentChatModel: null,       // full filename
    currentChatDisplay: 'Chat',   // short display name

    // Image (checkpoint) models
    imageModels: [],
    currentImageModel: null,
    currentImageDisplay: 'No Model',

    currentTheme: 'ember',
    availableThemes: ['ember', 'arctic', 'violet', 'jade', 'sunset'],

    // Chat history
    chatHistory: [],        // Array of { id, title, messages: [{role, text}], timestamp }
    activeChatId: null,     // Currently active chat ID

    // File attachments pending send
    pendingFiles: [],       // Array of { file: File, preview: dataURL|null, type: 'image'|'text' }

    // Web search mode
    searchMode: false       // When true, next message triggers web search
};

// CONSTANTS
const CONSTANTS = {
    API: {
        CHAT: '/chat',
        CHAT_STREAM: '/chat/stream',
        GENERATE: '/api/image/generate',
        MODELS: '/api/models',
        LLM_MODELS: '/api/llm-models',
        UPLOAD: '/api/upload',
        SEARCH: '/api/search'
    },
    THEMES_PATH: '/static/css/themes/',
    STORAGE_KEY_HISTORY: 'vox_chat_history',
    STORAGE_KEY_THEME: 'vox_theme'
};

// MODEL FAMILY PROFILES (auto-adjust defaults per model type)
const MODEL_PROFILES = {
    sd15:  { native: '512x512',  steps: 25, cfg: 7.0,  neg: true  },
    sd20:  { native: '768x768',  steps: 28, cfg: 7.5,  neg: true  },
    sdxl:  { native: '1024x1024', steps: 30, cfg: 5.5, neg: true  },
    flux:  { native: '1024x1024', steps: 20, cfg: 3.5, neg: false },
    pony:  { native: '1024x1024', steps: 25, cfg: 6.0, neg: true  },
    unknown: { native: '512x512', steps: 25, cfg: 7.0, neg: true  }
};

// --- INIT ---
window.onload = function () {
    loadChatHistory();
    initTheme();
    setupEventListeners();
    fetchAllModels();

    // Auto-resize textarea on load
    autoResize(document.getElementById('user-input'));

    // Initial UI Update
    updateUI();
    renderHistoryList();
};

function setupEventListeners() {
    // Navigation
    document.getElementById('nav-chat').onclick = () => switchMode('chat');
    document.getElementById('nav-image')?.addEventListener('click', () => switchMode('image'));
    document.getElementById('new-chat-btn').onclick = newChat;

    // Input
    const input = document.getElementById('user-input');
    input.oninput = () => autoResize(input);
    input.onkeydown = (e) => handleKey(e);
    document.getElementById('send-btn').onclick = handleSubmit;

    // Attachments
    const attachBtn = document.getElementById('attach-btn');
    const fileInput = document.getElementById('file-input');
    if (attachBtn && fileInput) {
        attachBtn.onclick = () => fileInput.click();
        fileInput.onchange = handleFileSelect;
    }

    // Search toggle
    const searchToggle = document.getElementById('search-toggle');
    if (searchToggle) {
        searchToggle.onclick = toggleSearchMode;
    }

    // Modals & Settings
    document.getElementById('model-selector').onclick = openModelModal;
    document.getElementById('settings-btn').onclick = toggleSettings;

    // Model modal click-outside handler
    const modelModal = document.getElementById('model-modal') || document.getElementById('modal-overlay');
    if (modelModal) {
        modelModal.onclick = (e) => {
            if (e.target === modelModal) closeModelModal();
        };
    }

    // Theme Selector (optional)
    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) themeSelect.onchange = (e) => setTheme(e.target.value);
}

// --- THEME ENGINE ---
function initTheme() {
    const saved = localStorage.getItem(CONSTANTS.STORAGE_KEY_THEME);
    if (saved && AppState.availableThemes.includes(saved)) {
        setTheme(saved);
    } else {
        setTheme('ember');
    }
}

function setTheme(themeName) {
    if (!AppState.availableThemes.includes(themeName)) return;
    AppState.currentTheme = themeName;
    localStorage.setItem(CONSTANTS.STORAGE_KEY_THEME, themeName);

    let link = document.getElementById('theme-stylesheet');
    const newHref = `${CONSTANTS.THEMES_PATH}${themeName}.css`;
    if (!link) {
        link = document.createElement('link');
        link.id = 'theme-stylesheet';
        link.rel = 'stylesheet';
        document.head.appendChild(link);
    }
    link.href = newHref;

    const dropdown = document.getElementById('theme-select');
    if (dropdown) dropdown.value = themeName;
}

// --- MODE SWITCHING ---
function switchMode(newMode) {
    AppState.mode = newMode;

    // Update Nav
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`nav-${newMode}`).classList.add('active');

    // Update Input Placeholder
    const input = document.getElementById('user-input');
    input.placeholder = newMode === 'chat'
        ? "Message VoxAI..."
        : "Describe the image you want to generate...";

    // Clear feed
    const feed = document.getElementById('chat-feed');
    feed.innerHTML = '';
    addSystemMessage(`Switched to ${newMode.toUpperCase()} mode.`);

    // Auto-show settings panel in image mode (neg prompt, resolution, etc.)
    const settingsPanel = document.getElementById('gen-settings');
    if (settingsPanel) {
        settingsPanel.style.display = newMode === 'image' ? 'block' : 'none';
    }

    // Update model display for the new mode
    updateUI();

    input.focus();
}

// --- CHAT HISTORY ---
function generateChatId() {
    return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
}

function loadChatHistory() {
    try {
        const raw = localStorage.getItem(CONSTANTS.STORAGE_KEY_HISTORY);
        AppState.chatHistory = raw ? JSON.parse(raw) : [];
    } catch (e) {
        AppState.chatHistory = [];
    }
}

function saveChatHistory() {
    try {
        // Keep max 50 chats
        if (AppState.chatHistory.length > 50) {
            AppState.chatHistory = AppState.chatHistory.slice(0, 50);
        }
        localStorage.setItem(CONSTANTS.STORAGE_KEY_HISTORY, JSON.stringify(AppState.chatHistory));
    } catch (e) {}
}

function saveCurrentChat() {
    // Collect messages from the DOM
    const feed = document.getElementById('chat-feed');
    const msgElements = feed.querySelectorAll('.message');
    if (msgElements.length === 0) return;

    const messages = [];
    msgElements.forEach(el => {
        const isUser = el.classList.contains('user-msg');
        const isAi = el.classList.contains('ai-msg');
        const content = el.querySelector('.msg-content');
        if (content && (isUser || isAi)) {
            messages.push({
                role: isUser ? 'user' : 'ai',
                html: content.innerHTML
            });
        }
    });

    if (messages.length === 0) return;

    // Get title from first user message text
    const firstUser = messages.find(m => m.role === 'user');
    const titleEl = document.createElement('div');
    titleEl.innerHTML = firstUser ? firstUser.html : 'Chat';
    let title = titleEl.textContent.trim().substring(0, 40);
    if (!title) title = 'Chat';

    if (AppState.activeChatId) {
        // Update existing
        const idx = AppState.chatHistory.findIndex(c => c.id === AppState.activeChatId);
        if (idx !== -1) {
            AppState.chatHistory[idx].messages = messages;
            AppState.chatHistory[idx].title = title;
            AppState.chatHistory[idx].timestamp = Date.now();
        }
    } else {
        // Create new
        const chatObj = {
            id: generateChatId(),
            title: title,
            messages: messages,
            timestamp: Date.now()
        };
        AppState.chatHistory.unshift(chatObj);
        AppState.activeChatId = chatObj.id;
    }

    saveChatHistory();
    renderHistoryList();
}

function loadChat(chatId) {
    // Save current first
    const feed = document.getElementById('chat-feed');
    if (feed.querySelectorAll('.message').length > 0 && AppState.activeChatId !== chatId) {
        saveCurrentChat();
    }

    const chat = AppState.chatHistory.find(c => c.id === chatId);
    if (!chat) return;

    feed.innerHTML = '';
    clearStats();
    AppState.activeChatId = chatId;

    chat.messages.forEach(m => {
        const div = document.createElement('div');
        div.className = `message ${m.role}-msg`;
        const avatarTxt = m.role === 'user' ? 'You' : 'V';
        div.innerHTML = `
            <div class="avatar">${avatarTxt}</div>
            <div class="msg-content">${m.html}</div>
        `;
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
        div.innerHTML = `
            <span style="overflow:hidden;text-overflow:ellipsis;">${chat.title}</span>
            <span class="delete-chat" onclick="deleteChat('${chat.id}', event)">✕</span>
        `;
        div.onclick = () => loadChat(chat.id);
        container.appendChild(div);
    });
}

function newChat() {
    // Save current chat if it has messages
    const feed = document.getElementById('chat-feed');
    if (feed.querySelectorAll('.message').length > 0) {
        saveCurrentChat();
    }

    // Clear and start fresh
    feed.innerHTML = '';
    clearStats();
    AppState.activeChatId = null;
    renderHistoryList();
    document.getElementById('user-input').focus();
}

// --- API LAYER ---
async function fetchAllModels() {
    // Fetch both LLM and image models in parallel
    const [llmResult, imgResult] = await Promise.allSettled([
        fetch(CONSTANTS.API.LLM_MODELS).then(r => r.json()),
        fetch(CONSTANTS.API.MODELS).then(r => r.json())
    ]);

    // LLM models
    if (llmResult.status === 'fulfilled') {
        AppState.chatModels = llmResult.value.models || [];
        if (AppState.chatModels.length > 0) {
            AppState.currentChatModel = AppState.chatModels[0].name;
            AppState.currentChatDisplay = AppState.chatModels[0].display;
        } else {
            AppState.currentChatModel = null;
            AppState.currentChatDisplay = 'Chat';
        }
    }

    // Image models
    if (imgResult.status === 'fulfilled') {
        AppState.imageModels = imgResult.value.models || [];
        if (AppState.imageModels.length > 0) {
            AppState.currentImageModel = imgResult.value.current || AppState.imageModels[0].name;
            AppState.currentImageDisplay = imgResult.value.current_display || AppState.imageModels[0].display;
        } else {
            AppState.currentImageModel = null;
            AppState.currentImageDisplay = 'No Model';
        }
    }

    updateUI();
}

// --- Get current model info based on mode ---
function getCurrentDisplay() {
    return AppState.mode === 'chat' ? AppState.currentChatDisplay : AppState.currentImageDisplay;
}

function getCurrentModelName() {
    return AppState.mode === 'chat' ? AppState.currentChatModel : AppState.currentImageModel;
}

function getCurrentModels() {
    return AppState.mode === 'chat' ? AppState.chatModels : AppState.imageModels;
}

// --- SUBMIT ---
async function handleSubmit() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text && AppState.pendingFiles.length === 0) return;

    // Show user message with attachment previews
    const attachHtml = buildAttachmentPreviewHtml();
    addMessage('user', text, attachHtml);
    input.value = '';
    input.style.height = 'auto';

    if (AppState.mode === 'chat') {
        // Upload files first, then send chat with file context
        let uploadedFiles = [];
        if (AppState.pendingFiles.length > 0) {
            uploadedFiles = await uploadFiles();
            clearAttachments();
        }
        await handleChat(text, uploadedFiles);
        // Auto-save after each exchange
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
        if (entry.type === 'image' && entry.preview) {
            html += `<img src="${entry.preview}" style="max-width:120px;max-height:80px;border-radius:6px;border:1px solid var(--border);" alt="${entry.file.name}">`;
        } else {
            html += `<span style="background:rgba(255,255,255,0.06);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:11px;">📄 ${entry.file.name}</span>`;
        }
    });
    html += '</div>';
    return html;
}

async function handleChat(text, uploadedFiles) {
    const typingId = addTyping();
    clearStats();

    // Build message with file context
    let fullMessage = text;
    if (uploadedFiles && uploadedFiles.length > 0) {
        const fileContext = uploadedFiles.map(f => {
            if (f.type === 'text') {
                return `[File: ${f.filename}]\n\`\`\`\n${f.content}\n\`\`\``;
            } else if (f.type === 'image') {
                return `[Image attached: ${f.filename} - ${f.description || 'User uploaded image'}]`;
            }
            return `[Attached: ${f.filename}]`;
        }).join('\n\n');

        fullMessage = fileContext + (text ? '\n\n' + text : '\n\nPlease analyze the attached files.');
    }

    // Check for web search intent (manual toggle or text pattern)
    const searchQuery = AppState.searchMode ? text : detectSearchIntent(text);
    if (searchQuery) {
        // Reset search mode after use
        if (AppState.searchMode) {
            AppState.searchMode = false;
            const searchBtn = document.getElementById('search-toggle');
            if (searchBtn) searchBtn.classList.remove('active');
            const inputEl = document.getElementById('user-input');
            if (inputEl) inputEl.placeholder = 'Message VoxAI...';
        }

        const results = await webSearch(searchQuery);
        if (results.length > 0) {
            addSearchResultCard(results, searchQuery);
            fullMessage = text + formatSearchResults(results);
        }
    }

    try {
        const resp = await fetch(CONSTANTS.API.CHAT_STREAM, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: fullMessage, model: AppState.currentChatModel })
        });

        if (!resp.ok) {
            // Fallback to non-streaming
            removeElement(typingId);
            const fallbackResp = await fetch(CONSTANTS.API.CHAT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: fullMessage, model: AppState.currentChatModel })
            });
            const data = await fallbackResp.json();
            addMessage('ai', data.reply || data.error || "No response received.");
            return;
        }

        // Streaming response
        removeElement(typingId);

        const feed = document.getElementById('chat-feed');
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-msg';
        msgDiv.innerHTML = `
            <div class="avatar">V</div>
            <div class="msg-content"><p>...</p></div>
        `;
        feed.appendChild(msgDiv);
        scrollToBottom();

        const contentEl = msgDiv.querySelector('.msg-content');
        let fullText = '';
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
                    const event = JSON.parse(dataStr);
                    if (event.type === 'chunk') {
                        fullText += event.text;
                        if (typeof marked !== 'undefined') {
                            contentEl.innerHTML = marked.parse(fullText);
                        } else {
                            contentEl.innerHTML = `<p>${fullText}</p>`;
                        }
                        scrollToBottom();
                    } else if (event.type === 'done') {
                        showStats(event.stats);
                    } else if (event.type === 'error') {
                        fullText += event.text;
                        contentEl.innerHTML = `<p>${fullText}</p>`;
                    }
                } catch (e) {}
            }
        }

        // Process remaining buffer
        if (buffer.startsWith('data: ')) {
            try {
                const event = JSON.parse(buffer.slice(6).trim());
                if (event.type === 'done') showStats(event.stats);
            } catch (e) {}
        }

    } catch (e) {
        removeElement(typingId);
        addMessage('ai', "[Error] Could not connect to AI Backend.");
    }
}

async function handleImage(prompt) {
    const loadingId = addTyping();

    // Parse resolution from preset dropdown
    const presetVal = document.getElementById('img-preset')?.value || '1024x1024';
    const [w, h] = presetVal.split('x').map(Number);

    const payload = {
        prompt: prompt,
        negative_prompt: document.getElementById('img-neg')?.value || '',
        width: w,
        height: h,
        steps: document.getElementById('img-steps')?.value || 20,
        cfg: document.getElementById('img-cfg')?.value || 7.0,
        model: AppState.currentImageModel
    };

    try {
        const resp = await fetch(CONSTANTS.API.GENERATE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await resp.json();
        removeElement(loadingId);

        if (data.status === 'success') {
            addImageCard(data.url, prompt, data.filename);
        } else {
            addMessage('ai', `[Image Error] ${data.error}`);
        }
    } catch (e) {
        removeElement(loadingId);
        addMessage('ai', `[System Error] ${e.message}`);
    }
}

// --- STATS / TOKENS PER SEC ---
function showStats(stats) {
    const el = document.getElementById('chat-stats');
    if (!el || !stats) return;
    el.textContent = `${stats.speed} t/s  ·  ${stats.tokens} tokens  ·  ${stats.duration}s`;
    el.style.display = 'block';
}

function clearStats() {
    const el = document.getElementById('chat-stats');
    if (el) {
        el.textContent = '';
        el.style.display = 'none';
    }
}

// --- DOM HELPERS ---
function updateUI() {
    const el = document.getElementById('current-model-name');
    if (el) el.innerText = getCurrentDisplay();
}

function addMessage(role, text, extraHtml) {
    const feed = document.getElementById('chat-feed');
    const div = document.createElement('div');
    div.className = `message ${role}-msg`;

    const avatarTxt = role === 'user' ? 'You' : 'V';
    const inner = role === 'ai' && typeof marked !== 'undefined'
        ? marked.parse(text)
        : `<p>${text}</p>`;

    div.innerHTML = `
        <div class="avatar">${avatarTxt}</div>
        <div class="msg-content">${inner}${extraHtml || ''}</div>
    `;
    feed.appendChild(div);
    scrollToBottom();
}

function addImageCard(url, prompt, filename) {
    const feed = document.getElementById('chat-feed');
    const cardHtml = `
        <div class="image-card">
            <img src="${url}" onclick="window.open('${url}', '_blank')">
            <div class="image-meta">
                <span>${prompt.substring(0, 40)}${prompt.length > 40 ? '...' : ''}</span>
                <a href="${url}" download="${filename}" class="download-link">Download</a>
            </div>
        </div>
    `;

    const div = document.createElement('div');
    div.className = 'message ai-msg';
    div.innerHTML = `
        <div class="avatar">IMG</div>
        <div class="msg-content" style="width:100%">
            <p>Generated Image:</p>
            ${cardHtml}
        </div>
    `;
    feed.appendChild(div);
    scrollToBottom();
}

function addTyping() {
    const id = 'typing-' + Date.now();
    const feed = document.getElementById('chat-feed');
    feed.insertAdjacentHTML('beforeend', `
        <div id="${id}" class="message ai-msg typing-anim" style="opacity:0.6;">
            <div class="avatar">...</div>
            <div class="msg-content"><p>Thinking...</p></div>
        </div>
    `);
    scrollToBottom();
    return id;
}

function addSystemMessage(text) {
    const feed = document.getElementById('chat-feed');
    feed.insertAdjacentHTML('beforeend', `
        <div style="text-align:center; opacity:0.5; font-size:12px; margin:20px 0;">
            ${text}
        </div>
    `);
}

function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    const area = document.getElementById('scroll-area');
    if (area) area.scrollTop = area.scrollHeight;
}

function autoResize(el) {
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
}

function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
    }
}

function toggleSettings() {
    const panel = document.getElementById('gen-settings');
    if (panel) {
        const isHidden = panel.style.display === 'none' || panel.style.display === '';
        panel.style.display = isHidden ? 'block' : 'none';
    }
}

// --- MODEL MODAL ---
function openModelModal() {
    const modal = document.getElementById('model-modal') || document.getElementById('modal-overlay');
    const list = document.getElementById('model-list-container');
    const title = document.getElementById('model-modal-title');
    if (!modal || !list) return;

    modal.style.display = 'flex';

    const models = getCurrentModels();
    const currentName = getCurrentModelName();
    const isChat = AppState.mode === 'chat';

    // Update modal title
    if (title) title.textContent = isChat ? 'Select LLM Model' : 'Select Image Model';

    list.innerHTML = '';

    if (models.length === 0) {
        list.innerHTML = `<div style="padding:20px;text-align:center;opacity:0.5;">
            No ${isChat ? 'LLM' : 'image'} models found
        </div>`;
        return;
    }

    models.forEach(m => {
        const div = document.createElement('div');
        const isSelected = m.name === currentName;
        div.className = `model-option ${isSelected ? 'selected' : ''}`;

        if (isChat) {
            // LLM model card
            div.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
                    <span style="font-weight:${isSelected ? '600' : '400'}">${m.display}</span>
                    <span style="font-size:11px; opacity:0.5;">${(m.size_gb || 0).toFixed(1)} GB</span>
                </div>
            `;
        } else {
            // Image model card
            div.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
                    <span style="font-weight:${isSelected ? '600' : '400'}">${m.display}</span>
                    <span style="font-size:11px; opacity:0.5;">${m.family || ''} · ${(m.size_gb || 0).toFixed(1)} GB</span>
                </div>
            `;
        }

        div.onclick = () => selectModel(m.name, m.display, m.family);
        list.appendChild(div);
    });
}

function closeModelModal() {
    const modal = document.getElementById('model-modal') || document.getElementById('modal-overlay');
    if (modal) modal.style.display = 'none';
}

function selectModel(name, display, family) {
    if (AppState.mode === 'chat') {
        AppState.currentChatModel = name;
        AppState.currentChatDisplay = display;
    } else {
        AppState.currentImageModel = name;
        AppState.currentImageDisplay = display;
        // Auto-adjust settings based on model family
        applyModelProfile(name, family);
    }
    updateUI();
    closeModelModal();
}

function applyModelProfile(modelName, knownFamily) {
    let family = knownFamily;
    if (!family) {
        // Detect family from model filename as fallback
        const lower = (modelName || '').toLowerCase();
        if (lower.includes('flux')) family = 'flux';
        else if (lower.includes('sdxl') || lower.includes('xl_') || lower.includes('_xl')) family = 'sdxl';
        else if (lower.includes('pony') || lower.includes('illustrious') || lower.includes('animagine')) family = 'pony';
        else if (lower.includes('sd2') || lower.includes('v2-') || lower.includes('768')) family = 'sd20';
        else family = 'sd15';
    }

    const profile = MODEL_PROFILES[family] || MODEL_PROFILES.unknown;

    // Update preset dropdown
    const preset = document.getElementById('img-preset');
    if (preset) preset.value = profile.native;

    // Update steps and CFG
    const steps = document.getElementById('img-steps');
    if (steps) steps.value = profile.steps;

    const cfg = document.getElementById('img-cfg');
    if (cfg) cfg.value = profile.cfg;

    // Show/hide negative prompt based on model support
    const neg = document.getElementById('img-neg');
    if (neg) {
        neg.parentElement.style.display = profile.neg ? '' : 'none';
        if (!profile.neg) neg.value = '';
    }
}

// --- FILE ATTACHMENTS ---
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    files.forEach(file => {
        const isImage = file.type.startsWith('image/');
        const entry = { file: file, preview: null, type: isImage ? 'image' : 'text' };

        if (isImage) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                entry.preview = ev.target.result;
                AppState.pendingFiles.push(entry);
                renderAttachPreview();
            };
            reader.readAsDataURL(file);
        } else {
            AppState.pendingFiles.push(entry);
            renderAttachPreview();
        }
    });

    // Reset file input so same file can be re-added
    e.target.value = '';
}

function renderAttachPreview() {
    const container = document.getElementById('attach-preview');
    if (!container) return;

    if (AppState.pendingFiles.length === 0) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    container.style.display = 'flex';
    container.innerHTML = '';

    AppState.pendingFiles.forEach((entry, idx) => {
        const chip = document.createElement('div');
        chip.className = 'attach-chip';

        if (entry.type === 'image' && entry.preview) {
            chip.innerHTML = `
                <img src="${entry.preview}" alt="preview">
                <span>${entry.file.name}</span>
                <span class="remove-attach" onclick="removeAttachment(${idx})">✕</span>
            `;
        } else {
            const ext = entry.file.name.split('.').pop().toUpperCase();
            chip.innerHTML = `
                <span style="font-size:16px;">📄</span>
                <span>${entry.file.name} <span style="opacity:0.4;font-size:10px;">${ext}</span></span>
                <span class="remove-attach" onclick="removeAttachment(${idx})">✕</span>
            `;
        }

        container.appendChild(chip);
    });
}

function removeAttachment(idx) {
    AppState.pendingFiles.splice(idx, 1);
    renderAttachPreview();
}

function clearAttachments() {
    AppState.pendingFiles = [];
    renderAttachPreview();
}

async function uploadFiles() {
    /** Upload pending files to server, returns array of {filename, type, content} */
    if (AppState.pendingFiles.length === 0) return [];

    const formData = new FormData();
    AppState.pendingFiles.forEach((entry, idx) => {
        formData.append('files', entry.file);
    });

    try {
        const resp = await fetch(CONSTANTS.API.UPLOAD, {
            method: 'POST',
            body: formData
        });
        if (!resp.ok) return [];
        const data = await resp.json();
        return data.files || [];
    } catch (e) {
        console.error('Upload failed:', e);
        return [];
    }
}

// --- WEB SEARCH ---
function toggleSearchMode() {
    AppState.searchMode = !AppState.searchMode;
    const btn = document.getElementById('search-toggle');
    const input = document.getElementById('user-input');
    if (btn) btn.classList.toggle('active', AppState.searchMode);
    if (input) {
        input.placeholder = AppState.searchMode
            ? 'Search the web...'
            : (AppState.mode === 'chat' ? 'Message VoxAI...' : 'Describe the image you want to generate...');
    }
}

async function webSearch(query) {
    /**
     * Search the web through IronGate's secure search proxy.
     * Returns array of {title, url, snippet}
     */
    try {
        const resp = await fetch(CONSTANTS.API.SEARCH, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, max_results: 5 })
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            const errMsg = errData.error || `Search returned status ${resp.status}`;
            addMessage('ai', `[Search Error] ${errMsg}`);
            return [];
        }
        const data = await resp.json();
        return data.results || [];
    } catch (e) {
        console.error('Search failed:', e);
        addMessage('ai', `[Search Error] Could not reach search service.`);
        return [];
    }
}

function detectSearchIntent(text) {
    /**
     * Detect if the user wants the AI to search the web.
     * Returns the search query if detected, null otherwise.
     * Must have both a search keyword AND a meaningful query (5+ chars).
     */
    const lower = text.toLowerCase().trim();

    // Too short to be a real search request
    if (lower.length < 10) return null;

    // Direct search commands: "search for X", "look up X", "google X"
    const directPatterns = [
        /^search\s+(?:for\s+)?(.+)/i,
        /^look\s+up\s+(.+)/i,
        /^google\s+(.+)/i,
        /^find\s+(?:info|information)\s+(?:on|about)\s+(.+)/i,
    ];

    for (const pat of directPatterns) {
        const m = text.match(pat);
        if (m) {
            const query = m[1].replace(/\?+$/, '').trim();
            if (query.length >= 5) return query;
        }
    }

    // Conversational: "can you search online for X", "could you look up X"
    const conversational = [
        /(?:can|could|would|please)?\s*(?:you\s+)?search\s+(?:online|the\s+web|the\s+internet)\s+(?:for\s+|and\s+(?:tell|find|show)\s+\w+\s+)(.+)/i,
        /(?:can|could|would|please)?\s*(?:you\s+)?search\s+for\s+(.+)/i,
        /(?:can|could|would|please)?\s*(?:you\s+)?look\s+up\s+(.+)/i,
    ];

    for (const pat of conversational) {
        const m = text.match(pat);
        if (m) {
            const query = m[1].replace(/\?+$/, '').trim();
            if (query.length >= 5) return query;
        }
    }

    // Keyword triggers: message contains "search online" or "search the web"
    if (lower.includes('search the web') || lower.includes('search online') || lower.includes('search the internet')) {
        // Strip the trigger phrase and use the rest as query
        const query = text.replace(/(?:can|could|would|please)?\s*(?:you\s+)?search\s+(?:online|the\s+web|the\s+internet)\s*(?:for|and)?\s*/i, '').replace(/\?+$/, '').trim();
        if (query.length >= 5) return query;
        return text.replace(/\?+$/, '').trim();
    }

    return null;
}

function formatSearchResults(results) {
    /** Format search results as context for the AI */
    if (!results.length) return '';

    let context = '\n\n---\n**Web Search Results:**\n';
    results.forEach((r, i) => {
        context += `\n${i + 1}. **${r.title}**\n   ${r.snippet}\n   Source: ${r.url}\n`;
    });
    context += '---\n\nPlease use the search results above to answer the question. Cite sources where relevant.';
    return context;
}

function addSearchResultCard(results, query) {
    /** Show search results in a nice card in the chat feed */
    const feed = document.getElementById('chat-feed');
    const div = document.createElement('div');
    div.className = 'message ai-msg';

    let resultsHtml = results.map((r, i) =>
        `<div style="margin:6px 0;padding:8px 10px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid var(--border);">
            <div style="font-size:13px;font-weight:500;color:var(--text-primary);">${r.title}</div>
            <div style="font-size:12px;color:var(--text-secondary);margin-top:3px;">${r.snippet}</div>
            <div style="font-size:11px;opacity:0.4;margin-top:3px;">${r.url}</div>
        </div>`
    ).join('');

    div.innerHTML = `
        <div class="avatar">🔍</div>
        <div class="msg-content">
            <p style="font-size:12px;opacity:0.6;margin-bottom:6px;">Search results for: "${query}"</p>
            ${resultsHtml}
        </div>
    `;
    feed.appendChild(div);
    scrollToBottom();
}
