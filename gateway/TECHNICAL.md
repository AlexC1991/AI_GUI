# Iron Tunnel v10.0 — Technical Specification

**Project:** Iron Tunnel  
**Version:** 10.0 (Client Generation Architecture)  
**Date:** February 2025

---

## Overview

Iron Tunnel is a secure gateway that allows you to share your local AI (Oobabooga, LM Studio, etc.) with specific people over the internet, without exposing raw IP/ports.

### Key Features

- **Host-Bound Clients**: Each generated client exe is cryptographically bound to YOUR specific host installation
- **One-Time Install Codes**: Like old game CD keys — use once, then it's consumed
- **Per-Session Approval**: Even with a valid client, each browser session requires a pairing code verification
- **Revocable Access**: Instantly kill any client exe remotely

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              HOST                                       │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   AI Backend    │◄───│   Iron Host     │◄───│     Ngrok       │     │
│  │  (Oobabooga)    │    │   (Gateway)     │    │    Tunnel       │     │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘     │
│                                  │                                      │
│                         ┌────────┴────────┐                             │
│                         │ Client Database │                             │
│                         │ - Client IDs    │                             │
│                         │ - Signatures    │                             │
│                         │ - Install Codes │                             │
│                         └─────────────────┘                             │
│                                  │                                      │
│                         Admin Console                                   │
│                         > generate "Alex"                               │
│                         > approve "Alex"                                │
│                         > revoke "Alex"                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ generate
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     GENERATED CLIENT EXE                                │
│                                                                         │
│  Hardcoded:                                                             │
│  - CLIENT_ID = "CLT-7F3A-X9K2-M4B8"                                     │
│  - SIGNATURE = "a8f3b2c1d4e5..." (64 chars)                             │
│  - BOUND_HOST = "HOST-541A-ER7X-9K2M"                                   │
│  - TUNNEL_URL = "https://xyz.ngrok.io"                                  │
│                                                                         │
│  Requires:                                                              │
│  - One-time install code to activate                                    │
│  - Network sniffer (Npcap) for traffic detection                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Security Model

### Layer 1: Host Identity

Each host installation generates a unique `HOST-XXXX-XXXX-XXXX` ID on first run. All generated clients are bound to this specific host. A client exe cannot work with a different host.

### Layer 2: Client Identity

Each generated client has:
- **Client ID**: Unique identifier (`CLT-XXXX-XXXX-XXXX`)
- **Signature**: 64-character cryptographic secret
- **Bound Host ID**: Must match the host it connects to

### Layer 3: One-Time Install Code

When you generate a client, you get an install code like `K7M2-X9R4-P3N8`. The user must enter this on first run to activate. Once used, the code is consumed — it cannot activate another installation.

### Layer 4: Session Pairing Code

Even after activation, each browser session requires approval:
1. User opens tunnel URL in browser
2. Client detects this and sends beacon with pairing code
3. Host shows: "Alex wants to connect. Code: ABCD1234"
4. Admin verifies code matches and approves
5. Session is active for configured duration (default 24h)

---

## Database Structure

### `host_identity.key`

```json
{
  "host_id": "HOST-541A-ER7X-9K2M",
  "created_at": "2025-02-05T10:30:00",
  "secret_key": "a8f3b2c1d4e5f6a7..."
}
```

### `clients.json`

```json
{
  "clients": {
    "CLT-7F3A-X9K2-M4B8": {
      "name": "Alex Laptop",
      "signature": "a8f3b2c1d4e5...",
      "install_code": "K7M2-X9R4-P3N8",
      "install_code_used": true,
      "bound_host": "HOST-541A-ER7X-9K2M",
      "created_at": "2025-02-05T10:35:00",
      "activated_at": "2025-02-05T14:22:00",
      "status": "ACTIVE"
    }
  }
}
```

### Client States

| State | Meaning |
|-------|---------|
| `PENDING_ACTIVATION` | Exe generated, install code not used yet |
| `ACTIVE` | Client is activated and valid |
| `REVOKED` | Admin killed it — exe is dead forever |

---

## Admin Commands

| Command | Description |
|---------|-------------|
| `generate "name"` | Create new client exe + install code |
| `list` | Show all clients |
| `sessions` | Show active browser sessions |
| `approve "name"` | Approve a session request |
| `kick "name"` | End active session (must re-approve) |
| `revoke "name"` | Permanently disable client exe |
| `lockdown` | Emergency: block ALL connections |
| `export` | Backup host identity |
| `status` | Overview |
| `clear` | Clear screen |

---

## User Flow

### For the Host Admin (You)

1. Run `iron_host.py` — generates Host ID on first run
2. Enter Ngrok token when prompted
3. Use `generate "Alex Laptop"` to create client
4. Give Alex:
   - The `.exe` file
   - The install code (e.g., `K7M2-X9R4-P3N8`)
5. When Alex connects, verify pairing code and `approve`

### For the Client User (Alex)

1. Receive `.exe` and install code from admin
2. Run exe (needs admin + Npcap on Windows)
3. Enter install code when prompted — one time only
4. Open tunnel URL in browser
5. Wait for admin to approve (show them your pairing code)
6. Once approved, use VoxAI chat interface

---

## Requirements

### Host

- Python 3.10+
- Ngrok account (free tier works)
- Local AI backend running (Oobabooga, etc.)
- PyInstaller (auto-installed for client generation)

### Generated Clients

- Windows: Npcap (https://npcap.com/)
- Run as administrator for packet capture
- One-time install code from host admin

---

## File Structure

```
iron_tunnel/
├── iron_host.py           # Main host application
├── ai_bridge.py           # AI backend connector
├── requirements.txt       # Python dependencies
├── start.bat              # Windows launcher
│
├── host_identity.key      # [Generated] Unique host ID
├── host_config.json       # [Generated] Settings
├── clients.json           # [Generated] Client database
│
├── templates/             # [Auto-created] HTML templates
│   ├── waiting.html
│   ├── voxai.html
│   └── expired.html
│
├── exports/               # [Generated] Client exe files
│   └── IronClient_*.exe
│
└── logs/                  # [Generated] Session logs
```

---

## Security Notes

### What This Protects Against

✅ Random scanners — they see nothing, get redirected  
✅ Unauthorized users — no client = no access  
✅ Stolen exe without install code — won't activate  
✅ Compromised client — revoke instantly  
✅ Session hijacking — pairing codes verify each session  

### What This Does NOT Protect Against

❌ Attacker with both exe AND install code before you use it  
❌ Compromised host machine  
❌ State-level adversaries  
❌ Physical access to activated client machine  

### Recommendations

1. **Send exe and install code separately** — different channels if possible
2. **Use short session durations** — limit exposure window
3. **Revoke promptly** — when access is no longer needed
4. **Backup host identity** — use `export` command
5. **Monitor sessions** — watch for unexpected requests

---

## Troubleshooting

### "Npcap required" on client

Install Npcap from https://npcap.com/ with "WinPcap API-compatible Mode" enabled.

### Client not detecting browser access

- Run client as administrator
- Ensure Npcap is installed
- Check if tunnel URL is correct

### "Wrong host" error on activation

Client exe was generated by a different host installation. Generate a new client from your host.

### PyInstaller build fails

- Ensure PyInstaller is installed: `pip install pyinstaller`
- Check antivirus isn't blocking the build
- Try running as administrator

---

*Iron Tunnel v10.0 — Secure AI Gateway*
