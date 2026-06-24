# Altagic Marketing OS — Product Requirements Document

**Date:** 2026-05-21  
**Author:** Rudra Dudhat  
**Status:** Draft — Awaiting Implementation Plan  
**Codename:** AMOS (Altagic Marketing Operating System)

---

## 1. Executive Summary

AMOS is an autonomous, multi-agent marketing intelligence system built for Altagic's in-house performance marketing team. It replaces the need for a senior marketing manager by delivering daily, role-specific creative briefs driven by real-time cultural signals. The system is human-in-the-loop by design — it generates intelligence, humans exercise creative judgment. Over time, it compounds brand-specific knowledge that no external tool can replicate.

**The core loop:**
```
Trend detected → Brand angle found → Role-specific brief delivered → 
Human executes → Performance logged → System learns → Next brief is smarter
```

---

## 2. Problem Statement

Altagic runs multiple in-house brands across clothing, home decor, and nutrition with a lean team of four. The marketing that actually works — pop-culture-first, funny, non-sell-forward content — requires:

- Real-time awareness of what's culturally relevant today
- Deep brand context to find the non-obvious intersection between trend and brand
- Role-specific guidance so each team member knows exactly what to do
- A system that gets smarter with use, not one that resets to zero every session

No off-the-shelf tool provides this. Content generators produce generic output. Marketing dashboards show data but give no direction. There is no tool that combines cultural intelligence, brand context, and role-specific briefing into a single daily workflow.

---

## 3. Product Vision

AMOS is not a content generator. It is a **brand intelligence that compounds over time**.

Month 1: The team gets daily briefs. Quality is good.  
Month 3: The system has absorbed 90 days of feedback, campaign results, and brand updates. Briefs are sharper than anything an external consultant could produce, because the context is proprietary.  
Month 6: The system knows Altagic's brands better than any new hire would in their first three months.

**Voice principle:** Every recommendation should feel like it came from a marketing manager who has watched too much Indian Twitter, knows the brand deeply, and has a sense of humor. Not formal. Not sell-forward. Pop-culture-first.

---

## 4. Users & Roles

| Role | Primary Need | Slack Channel |
|---|---|---|
| Influencer Marketing | Who to reach, with what angle | `#influencer-briefs` |
| Paid Ads | Which hooks are performing, ad copy variants | `#ads-briefs` |
| Video Editor | What format, length, trending audio, visual style | `#video-briefs` |
| Affiliate Marketing | Product angles, conversion-focused positioning | `#affiliate-briefs` |
| All Team | Shared daily cultural direction | `#daily-direction` |

There is no dedicated marketing manager. The system fills that role. Humans are the editorial gate — they approve by executing, redirect by replying in thread.

---

## 5. Core Architecture

### 5.1 Layer Overview

```
┌─────────────────────────────────────────────────────┐
│  EXTERNAL DATA LAYER (n8n cron jobs)                │
│  Google Trends · X/Twitter · Instagram · Apify      │
│  → Normalized → Redis cache (TTL: 24h)              │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│  INTELLIGENCE LAYER (Python)                        │
│                                                     │
│  Manager Agent                                      │
│  ├── Reads trend signals from Redis                 │
│  ├── Routes to brand sub-agents                     │
│  ├── Aggregates outputs                             │
│  └── Handles Slack conversation threads             │
│                                                     │
│  Brand Sub-Agents (one per brand)                   │
│  ├── Loads brand profile from Qdrant (RAG)          │
│  ├── Receives trend from Manager                    │
│  ├── Generates role-specific brief                  │
│  └── Returns structured JSON output                 │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│  OUTPUT LAYER                                       │
│  Slack API → role-specific channels                 │
│  Postgres → brief history + performance log         │
│  Langfuse → LLM observability + trace logging       │
└─────────────────────────────────────────────────────┘
```

### 5.2 Context Stores

| Store | What Lives Here | Why |
|---|---|---|
| **Qdrant** | Brand profiles, hook library, brief history | Semantic search — RAG retrieval at brief time |
| **Postgres** | Brief IDs, voting data, campaign performance, audit log | Relational — learning loop, attribution, idempotency |
| **Redis** | Today's trend cache, rate limits, idempotency keys | Fast, ephemeral — trend data is stale after 24h |

### 5.3 Agent Coordination

```
Manager Agent receives today's trends
    │
    ├── Routes trend to Brand Sub-Agent A (clothing)
    ├── Routes trend to Brand Sub-Agent B (home decor)
    └── Routes trend to Brand Sub-Agent C (nutrition)
         │
         Each sub-agent:
         1. Queries Qdrant for relevant brand context chunks
         2. Generates structured brief JSON
         3. Returns to Manager
         │
    Manager aggregates → routes to role-specific Slack channels
```

All sub-agents run in parallel via n8n queue mode.

### 5.4 Context Budget (hard limits per agent call)

| Component | Token Budget |
|---|---|
| Brand context (from Qdrant) | 800 tokens |
| Today's trend signal | 300 tokens |
| Task instruction | 200 tokens |
| Output format schema | 150 tokens |
| **Total** | **1,450 tokens** |

These are hard contracts, not guidelines. Tight input budgets keep agent behavior predictable and costs controlled. Never let prompts grow beyond this without a deliberate decision and schema update.

---

## 6. Brand Profile Schema

Every brand sub-agent loads a structured profile. Not prose — structured JSON.

```json
{
  "brand_id": "string",
  "name": "string",
  "niche": "clothing | home_decor | nutrition",
  "audience": {
    "age_range": "string",
    "primary_platform": "instagram | tiktok | youtube",
    "psychographic": "string"
  },
  "voice": {
    "is": ["adjective1", "adjective2", "adjective3"],
    "is_not": ["adjective1", "adjective2", "adjective3"]
  },
  "assets": {
    "team_details": "string",
    "collabs": ["string"],
    "behind_scenes_facts": ["string"]
  },
  "history": {
    "angles_that_worked": [{"angle": "string", "why": "string"}],
    "angles_that_flopped": [{"angle": "string", "why": "string"}]
  },
  "active_campaigns": ["string"],
  "hard_limits": ["string"],
  "last_updated": "ISO-8601 timestamp",
  "updated_by": "string"
}
```

---

## 7. Feature Specifications

### F1 — Trend Decay Timer
**What:** Every brief is tagged with a shelf life estimate — "this window closes in ~48 hours" vs. "7-day slow burn."  
**Why:** Helps the team triage what to ship immediately vs. what can wait.  
**How:** Trend detection layer measures velocity (rate of search/mention climb). Fast-climbing = short window. Plateau = longer window. Claude appends shelf life label to brief.  
**Slack output:** Tag on every brief: `⏱ 48h window` or `📅 7-day trend`

---

### F2 — Newsjack Alerts (Real-Time)
**What:** Separate from the morning brief. Immediate Slack push to `#daily-direction` when a trend spikes hard and fast.  
**Why:** Some moments (cockroach janta party energy) have a 4-hour window. The morning brief is too slow for these.  
**How:** n8n polling workflow checks trend velocity every 2 hours. If a trend crosses a spike threshold (configurable), triggers immediate alert across all brand channels.  
**Slack output:** `🚨 NEWSJACK WINDOW: [trend] — ~4h before this peaks. See brief below.`

---

### F3 — Competitor Breakdown — "Steal This"
**What:** When a competitor post goes viral, the system dissects it and posts a reframed version the brand can use without copying.  
**Why:** The fastest creative research loop — know what's working in the market before your team discovers it manually.  
**How:** Apify monitors 10–20 competitor accounts per brand. When a post exceeds an engagement threshold, triggers a Claude analysis: hook used, why it worked, non-copy reframe for your brand. Posts to relevant brand channel.  
**Slack output:** Posts to `#influencer-briefs` and `#ads-briefs`. Structured breakdown — original angle, why it worked, your version.

---

### F4 — Hook Library with Team Voting
**What:** Every suggested angle is stored. Team votes 👍/👎 in Slack. System builds a ranked hook library per brand over time.  
**Why:** The system compounds. After 30 days, briefs are drawn from patterns the team has already validated as resonant with their specific taste.  
**How:** Slack reaction event listener → Postgres stores vote + brief ID + brand ID + hook pattern text. A nightly job reads top-voted patterns from Postgres, embeds them, and upserts into Qdrant. At brief generation time, sub-agent retrieves top-voted hook patterns from Qdrant via semantic search against today's trend.

---

### F6 — Content Format Recommender
**What:** Not just what to post — how. "This trend performs 3x better as a 7-second Reel with [trending audio name] than as a carousel."  
**Why:** The video editor needs specific format guidance, not a topic to interpret from scratch.  
**How:** Trend signal includes platform performance data from the data layer. Claude maps trend type to format recommendation based on platform norms. Trending audio pulled from Instagram/TikTok trending audio trackers via Apify.  
**Slack output:** Appended to every video brief: `Format: 7s Reel · Audio: [name] · Aspect: 9:16`

---

### F7 — Brief → Ad Copy Converter
**What:** Paid ads person replies to any brief with `/convert-to-ad`. System outputs 3 Meta-ready copy variants: hook, body, CTA.  
**Why:** Cuts brief-to-launch time. The brief already has the angle — this is just a format transformation.  
**How:** Slack slash command → passes original brief + brand context to Claude with Meta ad copy prompt → returns 3 variants formatted for Facebook Ads Manager copy-paste.  
**Slack output:** 3 numbered variants, each with Hook / Body / CTA labeled.

---

### F8 — Influencer × Trend Matcher
**What:** Weekly Slack drop for the influencer person: 5 creators currently riding the dominant trend of the week, with engagement rate context.  
**Why:** Saves hours of manual discovery. The system surfaces creators who are already in the cultural moment, not creators who might fit.  
**How:** n8n weekly cron → Apify scrapes trending posts in niche → Claude filters for creators whose style matches brand voice + engagement signals in accessible range → formats into ranked list.  
**Slack output:** Posted every Monday to `#influencer-briefs` — creator handle, recent post that's trending, engagement stats, suggested angle for outreach.

---

### F11 — Weekly "What Worked" Digest
**What:** Every Monday morning, a team-wide summary: which angles won last week, what the dominant winning pattern was, what to double down on.  
**Why:** Closes the learning loop. Team sees that the system's recommendations translated to real results. Builds trust.  
**How:** n8n Monday cron → queries Postgres for last 7 days of voting data + any performance drops logged by team → Claude synthesizes into 5-bullet digest → posts to `#daily-direction`.  
**Slack output:** Simple 5-bullet format. No fluff. `✅ What landed · ❌ What flopped · 🔁 Double down on this week`

---

### F12 — Creator Brief Generator
**What:** Influencer person drops a creator's handle in Slack. System pulls their recent content, analyzes their style, and outputs a personalized outreach brief.  
**Why:** Cold outreach works when the pitch matches how the creator already talks. Generic briefs get ignored.  
**How:** Slack command → Apify scrapes creator's top 10 recent posts → Claude analyzes tone, format, recurring themes, audience energy → generates personalized outreach brief with a suggested opening line that mirrors their voice.  
**Slack output:** Creator analysis + brief with opening line + recommended angle.

---

### F16 — Platform-Specific Brief Variants
**What:** Every brief includes three variants of the same angle: Reels (7-second, punchy), X/Twitter (one-liner + thread potential), Stories (personal, interactive).  
**Why:** The team picks their platform for the day — the brief doesn't dictate it.  
**How:** Single Claude call with structured output prompt returning three format-specific variants from the same underlying angle. Adds ~200ms to generation time.  
**Slack output:** Three labeled sections per brief — `📱 Reels · 🐦 Twitter/X · 📖 Stories`

---

### F18 — Brand Context Update via Slack
**What:** Any team member types a natural update ("we just launched a collab with X" / "30% off hoodies this week") in `#brand-updates`. Relevant brand sub-agent absorbs it automatically.  
**Why:** Brand context must stay live. Manual prompt editing never happens. This makes updates effortless.  
**How:** Slack event listener on `#brand-updates` → Claude extracts structured context delta (what changed, which brand, effective dates) → writes to brand JSON in Qdrant with timestamp + author → audit log entry written to Postgres.  
**Constraint:** Every context write is logged. Timestamp + author + exact change. Auditable and reversible.

---

### F+ — Campaign Diagnostics Engine
**What:** Team drops a campaign idea into Slack. System returns: will this work, why/why not, how to reposition it for maximum engagement.  
**Why:** Replaces the "does this feel right?" conversation that currently happens in no one's head because there's no marketing manager.  
**How:** Slack command `/diagnose [campaign idea]` → Claude receives campaign brief + brand context + hook library history + current trend signal → returns structured diagnosis: Trend fit score, Brand fit score, Audience resonance score, Recommended repositioning if weak.  
**Response pattern:** Async — immediate "Analyzing..." reply, full diagnosis posted as thread reply (target: under 30 seconds, not guaranteed under load).  
**Slack output:** Scored breakdown + one concrete repositioning suggestion if any score is below threshold.

---

## 8. Security

### 8.1 Prompt Injection Prevention
Every piece of external content (scraped posts, trend data, competitor content) is wrapped in XML delimiters before reaching any prompt:

```xml
<external_untrusted_content source="competitor_post">
[content here]
</external_untrusted_content>
```

Claude treats content inside these tags as untrusted data, not instructions.

### 8.2 Principle of Least Privilege — Slack Bot
The bot has write access to exactly 6 channels:
- `#daily-direction`
- `#influencer-briefs`
- `#ads-briefs`
- `#video-briefs`
- `#affiliate-briefs`
- `#brand-updates` (read + write for context updates)

No workspace-wide read access. No DM access. If the bot token leaks, blast radius is limited to these channels.

### 8.3 Audit Log — Context Updates
Every brand profile write logs to Postgres:
```sql
context_audit (
  id, brand_id, changed_by, change_summary, 
  previous_value, new_value, timestamp
)
```
Any context degradation is traceable and reversible.

### 8.4 Secrets Management
All API keys (Claude, Slack, Apify, Qdrant) stored as environment variables injected at runtime. Never in code. Never in Slack messages. Rotate every 90 days.

### 8.5 Rate Limiting
Slash commands rate-limited at 10 calls per user per hour via Redis counter. Prevents accidental infinite loops and API cost blowouts.

---

## 9. Reliability

### 9.1 Graceful Degradation
| Failure | Fallback |
|---|---|
| Google Trends API down | Use Redis-cached trends from yesterday, tag brief with `⚠️ cached trend data` |
| Brand sub-agent fails | Post partial briefs for other brands, post alert for failed brand |
| Claude API timeout | Retry once with temperature 0.5. On second failure, post fallback message with yesterday's top angles |
| Apify scraper fails | Skip competitor/creator data for that run, continue without it |

The system never goes fully silent. Something always posts.

### 9.2 Idempotency
Every brief generation run gets a unique key: `brief-{brand_id}-{YYYY-MM-DD}`. Before generating, check Postgres for this key. If found, skip. Prevents double-posting if n8n cron fires twice.

### 9.3 Output Validation
Claude's output is parsed against a strict Pydantic schema before posting to Slack. If validation fails: retry once. If it fails twice: post fallback with error tag. Malformed output never reaches the team.

### 9.4 Context Freshness Enforcement
If `last_updated` on a brand profile is >30 days old, the brief for that brand prepends:  
`⚠️ Brand context is 30+ days old — outputs may not reflect current positioning. Update via #brand-updates.`

### 9.5 Observability
Every agent call traced in Langfuse:
- Which prompt version
- Which brand
- Which trend input
- Token count
- Latency
- Output (stored for comparison)

When brief quality drops, the trace in Langfuse shows exactly which input or prompt change caused it.

---

## 10. Scalability

### 10.1 Stateless Agents
Agents hold zero state in memory. All state in external stores. Any agent instance can be killed, restarted, or scaled horizontally without data loss or behavioral change.

### 10.2 Parallel Brand Processing
All brand sub-agents run in parallel via n8n queue mode. Adding a new brand = adding a new brand profile to Qdrant. Zero architectural changes.

### 10.3 RAG over Context Injection
Brand context lives in Qdrant as chunked, embedded documents. At brief generation time, only the chunks semantically relevant to today's trend are retrieved. As brand profiles grow (months of updates), the context quality improves without hitting token limits.

### 10.4 Schema-First Everything
Brand profiles, brief outputs, voting records, audit logs — all defined as strict schemas before any code is written. Schemas are the contract between components. Changing a schema requires a migration, not a hotfix.

---

## 11. Daily Adoption Strategy

**Zero friction delivery.** Brief arrives in Slack at 8:00am without anyone triggering it. No login, no dashboard, no link to click. It is simply there.

**Visibly smarter over time.** After 3 weeks, the system posts to `#daily-direction`: "Based on your team's feedback, [X hook pattern] outperforms [Y] by 2x for Brand Z. Today's briefs reflect this." The team sees the loop closing.

**Attribute wins back to the brief.** When campaign performance data is logged, the system traces it back to the originating brief and posts: "This angle (from Monday's brief) contributed to this result." Attribution builds trust faster than any feature.

**Effortless pushback.** A 👎 on any brief triggers a follow-up in thread: `What missed? [Wrong trend] [Wrong tone] [Wrong brand fit]`. One tap. Logged. The team feels heard.

**Conversational thread = retention mechanism.** The quality of "why did you suggest this?" answers is as important as brief quality. Every thread response from the bot must be specific, reasoned, and honest about uncertainty.

---

## 12. MVP Scope (Phase 1)

Build these first. Validate with the team. Then expand.

| Feature | Phase |
|---|---|
| Daily brief generation (manager + brand sub-agents) | MVP |
| Role-specific Slack channel delivery | MVP |
| Platform-specific variants (F16) | MVP |
| Brand context update via Slack (F18) | MVP |
| Trend Decay Timer (F1) | MVP |
| Newsjack Alerts (F2) | MVP |
| Hook Library + Voting (F4) | MVP |
| Campaign Diagnostics Engine (F+) | MVP |
| Brief → Ad Copy Converter (F7) | Phase 2 |
| Competitor Breakdown "Steal This" (F3) | Phase 2 |
| Creator Brief Generator (F12) | Phase 2 |
| Influencer × Trend Matcher (F8) | Phase 2 |
| Content Format Recommender (F6) | Phase 2 |
| Weekly "What Worked" Digest (F11) | Phase 2 |

Phase 1 validates the core loop. Phase 2 adds the intelligence that makes it irreplaceable.

---

## 13. Success Metrics

| Metric | Target (30 days) |
|---|---|
| Daily brief open rate (proxy: thread engagement) | >80% of briefs get at least one team reaction |
| Hook library size | >50 rated hooks per brand |
| Brief-to-execution rate | >60% of briefs lead to a campaign that week |
| Context update frequency | At least 2 brand context updates per week |
| Team NPS on brief quality | >7/10 (informal weekly check-in) |

The system is working when the team would feel its absence on a day it didn't run.

---

## 14. Tech Stack Summary

| Component | Tool |
|---|---|
| Agent intelligence | Python + Claude API (claude-sonnet-4-6) |
| Workflow orchestration | n8n (queue mode) |
| Slack interface | Slack Bolt (Python) |
| Vector store | Qdrant |
| Relational store | Postgres |
| Cache + rate limiting | Redis |
| Web scraping | Apify |
| LLM observability | Langfuse |
| Trend data | Google Trends API + X/Twitter API + Apify Instagram |

---

*End of PRD. Next step: implementation plan.*
