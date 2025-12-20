# Environment Variables - Secure Setup Guide

## Overview

This project uses **two separate `.env` files** for security:

1. **Root `.env`** - Shared, safe variables (accessible to frontend)
2. **`backend_ml/.env.local`** - Backend-only secrets (never exposed to frontend)

## Why This Approach?

### Security Concern
If we put everything in one root `.env` file, there's a risk that sensitive backend credentials (like `SUPABASE_SERVICE_KEY` with admin privileges) could accidentally be exposed to the frontend bundle.

### Solution
- **Frontend** only accesses the root `.env` with public-safe values
- **Backend** reads from BOTH files, keeping secrets in `backend_ml/.env.local`

## Setup Instructions

### 1. Root `.env` (Shared Variables)

Create `/nba-holistic-predictor/.env`:

```bash
# Copy the example file
cp .env.example .env
```

Fill in these **PUBLIC-SAFE** values:

```bash
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGc...your-anon-key
```

✅ **Safe for frontend** - These are designed to be public
✅ Loaded by: Frontend (Vite) + Backend (Python)

---

### 2. Backend `.env.local` (Backend Secrets)

Create `/nba-holistic-predictor/backend_ml/.env.local`:

```bash
cd backend_ml
cp .env.local.example .env.local
```

Fill in these **SECRET** values:

```bash
SUPABASE_SERVICE_KEY=eyJhbGc...your-SERVICE-key
NBA_API_KEY=your-api-key-if-needed
```

❌ **Never expose to frontend** - Admin privileges!
✅ Loaded by: Backend (Python) only

---

## How It Works

### Frontend (Vite/React)
```javascript
// frontend_web/src/supabaseClient.js
const supabaseUrl = import.meta.env.SUPABASE_URL;        // ✅ From root .env
const supabaseAnonKey = import.meta.env.SUPABASE_ANON_KEY; // ✅ From root .env
```

**Vite Configuration:**
- `envDir: '../'` tells Vite to look in the root directory
- Only reads from root `.env`
- Cannot access `backend_ml/.env.local`

### Backend (Python)
```python
# backend_ml/any_module.py
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

# SUPABASE_URL from root .env
# SUPABASE_SERVICE_KEY from backend_ml/.env.local
```

**Python Configuration (`backend_ml/config.py`):**
- Loads root `.env` first
- Then loads `backend_ml/.env.local` (secrets override if conflicts)
- Both files are merged safely

---

## File Locations Summary

```
nba-holistic-predictor/
├── .env                          ✅ Shared, public-safe (in .gitignore)
├── .env.example                  📄 Template for root .env
│
└── backend_ml/
    ├── .env.local                ❌ Secrets only (in .gitignore)
    ├── .env.local.example        📄 Template for backend secrets
    └── config.py                 🔧 Loads from both files
```

---

## Security Checklist

- [x] Root `.env` is in `.gitignore` (never committed)
- [x] `backend_ml/.env.local` is in `.gitignore` (never committed)
- [x] Only public `SUPABASE_ANON_KEY` in root `.env`
- [x] Secret `SUPABASE_SERVICE_KEY` in `backend_ml/.env.local`
- [x] Frontend cannot access `backend_ml/.env.local`
- [x] Example files (`.env.example`, `.env.local.example`) are safe to commit

---

## What Goes Where?

| Variable | Root `.env` | `backend_ml/.env.local` | Frontend Access | Backend Access |
|----------|-------------|-------------------------|----------------|----------------|
| `SUPABASE_URL` | ✅ | - | ✅ Yes | ✅ Yes |
| `SUPABASE_ANON_KEY` | ✅ | - | ✅ Yes | ✅ Yes |
| `SUPABASE_SERVICE_KEY` | ❌ | ✅ | ❌ No | ✅ Yes |
| `NBA_API_KEY` | ❌ | ✅ | ❌ No | ✅ Yes |

---

## Quick Start

```bash
# 1. Set up root .env (shared, safe variables)
cp .env.example .env
# Edit .env and add SUPABASE_URL and SUPABASE_ANON_KEY

# 2. Set up backend secrets
cd backend_ml
cp .env.local.example .env.local
# Edit .env.local and add SUPABASE_SERVICE_KEY

# 3. Verify they're not tracked by git
cd ..
git status  # Should NOT show .env or .env.local
```

---

## Common Questions

**Q: Can I put everything in one file?**
A: Not recommended. While Vite won't expose variables without `VITE_` prefix, it's safer to physically separate secrets.

**Q: What if I forget to create `.env.local`?**
A: The backend will still work for reading data (using `SUPABASE_ANON_KEY`), but admin operations requiring `SUPABASE_SERVICE_KEY` will fail.

**Q: Can I use a different name than `.env.local`?**
A: Yes, just update `backend_ml/config.py` to point to your file name.

---

**Security First:** When in doubt, keep secrets in `backend_ml/.env.local` and out of the root `.env`.
