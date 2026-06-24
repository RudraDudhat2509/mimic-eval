# Formie — Design Spec
**Date:** 2026-06-09  
**Status:** Approved  
**Phase:** v1

---

## 1. Product Vision

Formie is a Chrome extension that acts as an intelligence layer on top of Google Forms. It replaces the slow, click-heavy form-building experience with natural language — you describe what you want, Formie builds it. The mascot is Blanc, a minimal pink cloud cat that lives in the bottom-right corner of every Google Forms page (and every other webpage for the select-text feature).

The goal for v1: something genuinely good enough that engineers inspect it, and viral enough that people share it on Twitter. Built to production quality, not demo quality.

---

## 2. Architecture

**Type:** Chrome Manifest V3 extension  
**Backend:** None — pure client-side  
**Storage:** `chrome.storage.local`  
**LLM:** Direct API calls from content script  
**Forms:** Google Forms API via `chrome.identity` OAuth

```
chrome-extension/
├── manifest.json
├── service-worker.js          # install event, storage defaults
├── popup/
│   ├── popup.html             # settings: API keys, memory reset, Slack webhook
│   └── popup.js
├── content/
│   ├── formie.js              # injected on docs.google.com/forms — builder mode
│   ├── selector.js            # injected on ALL pages — select-text feature
│   └── blanc.js               # Blanc widget rendering + animation state machine
├── llm/
│   ├── cascade.js             # provider fallback logic
│   ├── providers/
│   │   ├── gemini.js
│   │   ├── grok.js
│   │   └── ollama.js
│   └── prompts.js             # all system prompts in one place
├── forms/
│   └── google-forms-api.js    # Google Forms API wrapper
├── memory/
│   └── store.js               # read/write chrome.storage.local
├── templates/
│   └── index.js               # built-in templates + template matching logic
├── integrations/
│   └── slack.js               # outbound webhook
└── assets/
    └── blanc.svg              # the cat
```

**Why no backend:**  
Zero hosting cost, zero breach risk, ships faster. User data never leaves their machine. Stated as a feature: *"your forms, your data, your device."*

---

## 3. Features

### 3.1 Natural Language Form Builder

**Trigger:** User is on `docs.google.com/forms/d/*/edit` — Blanc auto-appears.

**Flow:**
1. User clicks Blanc (or the base platform)
2. Prompt input slides up: `"describe your form..."`
3. User submits → Blanc enters Thinking state
4. LLM cascade generates form schema as JSON
5. Questions stream into Google Forms one by one (async generators)
6. Blanc enters Done state, shows snarky quip
7. Form schema saved to memory

**Blanc states:**
| State | Visual | Trigger |
|---|---|---|
| Idle | Floating, batting yarn ball | Default on page load |
| Listening | Yarn disappears, eyes wide, prompt input visible | User clicks |
| Thinking | Eyes → `...` dots, tail wags fast | Prompt submitted |
| Done | Happy blink, yarn returns, quip fades in | Form created |
| Error | Head tilt, specific error quip | Any failure |

---

### 3.2 Streaming Form Generation (Async Generators)

Form questions stream into Google Forms in real time as the LLM generates them. The user sees questions appearing one by one rather than waiting for the full form to appear at once.

**Why async generators here:**  
Gemini Flash supports streaming responses. Instead of waiting for the complete JSON, we parse the stream chunk-by-chunk and yield each question as it arrives. This is both faster perceived performance and a better visual — Blanc looks alive and working, not frozen.

```javascript
async function* streamFormQuestions(prompt, provider) {
  const stream = await provider.streamGenerate(SYSTEM_PROMPT, prompt);
  let buffer = '';

  for await (const chunk of stream) {
    buffer += chunk;
    const questions = tryParsePartialQuestions(buffer);
    for (const q of questions) {
      yield q; // each question yielded as soon as it's parseable
    }
  }
}

// Caller:
for await (const question of streamFormQuestions(userPrompt, activeProvider)) {
  await googleFormsApi.addQuestion(formId, question);
  blanc.showQuestionAdded(question.title); // real-time Blanc feedback
}
```

---

### 3.3 Select Any Text → Create Form

**Trigger:** User selects text on **any webpage** (not just Google Forms).

**Flow:**
1. `selector.js` runs on all pages, listens for `mouseup`
2. If selected text > 20 chars, a small Blanc bubble appears near the selection
3. Bubble shows: `"turn this into a form?"`
4. User clicks → new tab opens with Google Forms → form created from selected text
5. Blanc pops in the new tab, shows: `"built from what you selected. nice taste."`

**What this enables:**
- Select requirements from a spec doc → instant form
- Select interview questions from a doc → survey form
- Select a list of tasks → task tracker form
- Select a WhatsApp-style conversation → extract questions from it

**Performance note:**  
`selector.js` is lightweight — it only activates `mouseup` and renders a small absolutely-positioned div. No heavy processing until the user clicks.

---

### 3.4 Smart Template Library

**Two types of templates:**

**Built-in (10 pre-installed):**
| Template | Fields |
|---|---|
| Event RSVP | Name, Email, Attending, Dietary needs, +1 |
| Job Application | Name, Email, Phone, Role, LinkedIn, Cover note |
| Feedback Form | Name (optional), Rating (1-5), What went well, What to improve |
| Bug Report | Title, Steps to reproduce, Expected vs actual, Severity |
| College Application | Name, DOB, Course, CGPA, SOP, Documents |
| Company Onboarding | Name, Role, Start date, Equipment needed, Emergency contact |
| Quiz | Title, Questions (MCQ), Pass mark, Show results |
| Customer Survey | NPS score, Product feedback, Feature requests |
| Meeting Feedback | Meeting title, Presenter, Clarity (1-5), Action items |
| Registration Form | Name, Email, Phone, Organization, T-shirt size |

**User-saved templates:**  
After every form creation, Blanc offers: `"save this as a template?"`. User names it. Stored in `chrome.storage.local`.

**Context-aware matching:**  
When a user types a prompt, Formie scores it against all templates (built-in + saved) using keyword matching + LLM classification. If a match is found above threshold, Blanc says: `"this looks like a [Job Application]. use my template or build from scratch?"`

The LLM gets told which template matched so it can adapt field names and structure to the user's specific prompt while keeping the template's structure as a scaffold.

---

### 3.5 Form Modification via Prompt

**Trigger:** User is on an existing Google Form edit page and types a modification prompt.

**Examples:**
- `"add a phone number field after question 2"`
- `"make all questions required"`
- `"add a section break before the last question"`
- `"change question 3 to a dropdown"`

**How it works:**
1. Formie reads the current form structure via Google Forms API (`GET /v1/forms/{formId}`)
2. Sends current structure + modification prompt to LLM
3. LLM returns a `batchUpdate` diff — only the changes, not the full form
4. Formie applies the diff via `POST /v1/forms/{formId}:batchUpdate`

This is non-destructive — existing questions are never deleted unless explicitly told to.

---

### 3.6 Form Completeness Checker

After every form generation, Formie runs a silent check:

```javascript
const COMMON_MISSING = {
  contact: ['email', 'phone', 'contact'],
  identity: ['name', 'full name'],
  date: ['date', 'when', 'time'],
};
```

If the generated form is missing a field category that's typical for its type (e.g., a registration form with no email field), Blanc flags it:

`"you might want a contact field — want me to add one?"`

One-click to add. One-click to dismiss. Never blocks the flow.

---

### 3.7 LLM Fallback Cascade

**Provider chain** (tried in order):
```
1. Gemini Flash   — primary, free tier, BYOK
2. Grok (xAI)    — fallback 1, optional BYOK
3. Ollama local  — fallback 2, auto-detected at localhost:11434
```

**Error handling per failure type:**

| Error | Blanc says | Action |
|---|---|---|
| Auth failed | `"[provider] key is wrong. check settings."` | Skip to next provider |
| Rate limited | *(silent)* | Skip to next provider |
| Parse failed | *(silent, retry once)* | Retry same provider, then skip |
| Network error | *(silent)* | Skip to next provider |
| All failed | `"nothing's responding. check your keys."` | Show settings button |

**Provider interface** (all providers implement the same shape):
```javascript
// Every provider exposes the same interface.
// Adding a new provider = adding one file. The cascade loop doesn't change.
{
  name: 'gemini',
  isAvailable: async () => Boolean,
  generate: async (system, user) => String,
  streamGenerate: async function*(system, user) // async generator
}
```

---

### 3.8 Memory / Context Pool

Stored in `chrome.storage.local`:

```javascript
{
  keys: {
    gemini: String,
    grok: String,       // optional
    slackWebhook: String // optional
  },
  memory: {
    fieldSchemas: [
      // auto-saved after each form build
      { name: String, fields: [String], usageCount: Number, lastUsed: Date }
    ],
    history: [
      // last 20 forms
      { title: String, questionCount: Number, templateUsed: String|null, createdAt: Date }
    ],
    savedTemplates: [
      // user-named templates
      { name: String, schema: FormSchema, createdAt: Date }
    ]
  }
}
```

**Memory injection into prompts:**  
When memory has ≥3 entries, Formie appends a context block to every system prompt:

```
User's frequent field patterns: [Name, Email, Role, Company]
Recent forms: ["Event RSVP", "Q3 Feedback", "Bug Report"]
Adapt field naming and structure to match their style.
```

---

### 3.9 Slack Outbound Integration

After a form is created, Blanc shows:

`"share to slack? 🔗"`

If user clicks → Formie POSTs to the configured incoming webhook URL:

```json
{
  "text": "📋 *New form created by Formie*",
  "blocks": [
    {
      "type": "section",
      "text": { "type": "mrkdwn", "text": "*Event Feedback Form*\n6 questions • Created just now" }
    },
    {
      "type": "actions",
      "elements": [
        { "type": "button", "text": { "type": "plain_text", "text": "Open Form" }, "url": "<form_edit_url>" },
        { "type": "button", "text": { "type": "plain_text", "text": "Share with Respondents" }, "url": "<form_view_url>" }
      ]
    }
  ]
}
```

Setup: user pastes their Slack incoming webhook URL in the popup settings page. One field, one paste, done.

---

## 4. Onboarding Flow

```
Install extension
       ↓
Open any Google Forms page
       ↓
Blanc appears — bubble: "hey. i'm Formie.
                          drop your Gemini API key and i'll do the rest.
                          → aistudio.google.com (it's free)"
       ↓
User pastes key → Formie pings Gemini with a test request
       ↓
Success → bubble: "perfect. describe a form. i'll build it."
Failure → bubble: "that key didn't work. try again."
       ↓
User builds first form → template offered for saving → schema saved to memory
       ↓
Every form after: memory context injected, Blanc gets smarter
```

---

## 5. LLM System Prompt

```
You are a Google Forms builder. Output ONLY valid JSON — no markdown, no explanation.

Schema:
{
  "title": "string",
  "description": "string",
  "questions": [
    {
      "title": "string",
      "type": "SHORT_ANSWER | PARAGRAPH | MULTIPLE_CHOICE | CHECKBOX | DROPDOWN | LINEAR_SCALE | DATE",
      "required": true | false,
      "options": ["string"],           // MULTIPLE_CHOICE, CHECKBOX, DROPDOWN only
      "scale": {                       // LINEAR_SCALE only
        "low": 1, "high": 5,
        "lowLabel": "", "highLabel": ""
      }
    }
  ]
}

Rules:
- 4–8 questions unless user specifies a number
- Pick the most appropriate field type for each question
- required: true for identity/contact fields, false for optional feedback
- Never return anything except the JSON object
- If a template is provided, use it as structural scaffolding but adapt to the user's prompt

{{MEMORY_CONTEXT}}
{{TEMPLATE_CONTEXT}}
```

---

## 6. Google Forms API Integration

**OAuth scope required:** `https://www.googleapis.com/auth/forms.body`

Obtained via `chrome.identity.getAuthToken({ interactive: true })` — uses the user's existing Google login, no separate sign-in screen after the first prompt.

**API calls used:**
| Operation | Endpoint |
|---|---|
| Create form | `POST /v1/forms` |
| Add questions | `POST /v1/forms/{id}:batchUpdate` |
| Read structure | `GET /v1/forms/{id}` |
| Modify form | `POST /v1/forms/{id}:batchUpdate` |

---

## 7. Blanc Widget Spec

- **Position:** Fixed, bottom-right, `z-index: 999999`
- **Size:** 130px wide widget, ~200px tall cat SVG
- **Design:** Minimal pink cloud blob (Blanc), thick black outlines, white fill with pink tint, heart nose, shine-dot eyes
- **Base:** Small white pill/platform Blanc sits on
- **Idle animation:** Gentle float, yarn batting, ear twitch, slow blink
- **Speech bubble:** Slides up from above Blanc, max 160px wide, fades out after 5s
- **Prompt input:** Appears inside the speech bubble area on click
- **Responsive:** Never overlaps form content — shifts left on narrow viewports

---

## 8. Phase 2 Roadmap (not in scope for v1)

- Smart autofill from context pool (filler mode)
- Bulk response import (CSV/sheets → Google Forms responses)
- Conditional logic generation
- Aesthetic theme picker
- Voice input
- Multi-language form generation
- Drop-off predictor
- Expand to other form domains (Typeform, JotForm, Airtable)
- Slack slash command (`/formie create ...`) — requires Cloudflare Worker

---

## 9. Tech Stack Summary

| Layer | Choice | Why |
|---|---|---|
| Extension | Chrome MV3 | Required for Chrome Web Store 2024+ |
| LLM primary | Gemini 1.5 Flash | Free tier, fast, BYOK |
| LLM fallback | Grok / Ollama | Production reliability |
| Forms API | Google Forms REST API | Reliable vs DOM manipulation |
| Auth | `chrome.identity` | Silent OAuth, user already logged in |
| Storage | `chrome.storage.local` | No backend, no breach risk |
| Async patterns | Async generators | Streaming form generation |
| Slack | Incoming webhooks | No backend required |
