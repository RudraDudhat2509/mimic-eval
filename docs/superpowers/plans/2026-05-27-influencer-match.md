# InfluencerMatch AI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph pipeline that reads a campaign brief, discovers ~80 influencer candidates, scores them dynamically, enriches the top 20 with GPT-4o-generated campaign ideas and outreach messages, deduplicates against Google Sheets, writes results to Sheets, and posts a Slack summary — all triggerable via FastAPI for n8n.

**Architecture:** Five LangGraph nodes run sequentially (discover → enrich → score → intelligence → output). Discovery calls Apify's `instagram-hashtag-scraper` actor; enrichment calls `instagram-profile-scraper`. Mock scraper is kept for unit tests only — the graph always uses Apify in production. Deduplication is done by checking existing usernames in the "All" Google Sheets tab before writing. FastAPI exposes a single `POST /run` endpoint that n8n calls via HTTP Request node on a daily schedule.

**Tech Stack:** Python 3.11, LangGraph 0.1.x, LangChain, OpenAI GPT-4o, Apify (`apify-client`) for Instagram scraping, gspread + google-auth, FastAPI, Uvicorn, Pydantic v2, PyYAML, python-dotenv, pytest

---

## File Structure

```
influencer-match/
├── main.py                    # FastAPI app — POST /run, GET /health
├── config/
│   └── campaign.yaml          # Campaign brief (edit per brand)
├── models/
│   └── schemas.py             # Pydantic: CampaignBrief, RawProfile, ScoredInfluencer, PipelineState
├── scrapers/
│   ├── mock.py                # Fake profile generator — used for unit tests only
│   └── apify_scraper.py       # Real scraper: Apify instagram-hashtag-scraper + instagram-profile-scraper
├── agents/
│   ├── graph.py               # LangGraph StateGraph — wires all nodes
│   ├── discovery.py           # Finds candidate usernames (calls mock or instagram scraper)
│   ├── scoring.py             # Dynamic scoring engine — pure Python, no LLM
│   └── intelligence.py        # GPT-4o: campaign idea + outreach message per influencer
├── output/
│   ├── sheets.py              # gspread writer + dedup check against "All" tab
│   └── slack.py               # Slack webhook — posts daily summary
├── tests/
│   ├── test_scoring.py        # Unit tests for every scoring sub-function
│   ├── test_dedup.py          # Unit test for sheets dedup logic (mocked gspread)
│   └── test_pipeline.py       # Integration test: full graph run with mocked GPT-4o
├── .env.example               # Template — OPENAI_API_KEY, SLACK_WEBHOOK_URL, GOOGLE_SHEET_ID
├── service_account.json       # Google service account credentials (gitignored)
└── requirements.txt
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `influencer-match/requirements.txt`
- Create: `influencer-match/.env.example`
- Create: `influencer-match/config/campaign.yaml`
- Create: all `__init__.py` files in each package

- [ ] **Step 1: Create root directory and subdirectories**

```bash
mkdir -p influencer-match/{config,models,scrapers,agents,output,tests}
touch influencer-match/{models,scrapers,agents,output,tests}/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

```
# influencer-match/requirements.txt
fastapi==0.111.0
uvicorn==0.29.0
langgraph==0.1.13
langchain==0.2.0
langchain-openai==0.1.7
openai==1.30.0
pydantic==2.7.1
gspread==6.1.2
google-auth==2.29.0
httpx==0.27.0
pyyaml==6.0.1
python-dotenv==1.0.1
instagrapi==2.1.2
pytest==8.2.0
pytest-asyncio==0.23.7
```

- [ ] **Step 3: Write .env.example**

```
# influencer-match/.env.example
OPENAI_API_KEY=sk-...
APIFY_API_TOKEN=apify_api_xxxx
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
GOOGLE_SHEET_ID=your_sheet_id_here
USE_MOCK_SCRAPER=false   # set to true to skip Apify calls in local dev/tests
```

- [ ] **Step 4: Write config/campaign.yaml**

```yaml
# influencer-match/config/campaign.yaml
brand_name: "Perfect Plants"
product_type: "indoor plants"
target_audience: "urban millennials"
niches:
  - gardening
  - home decor
follower_range: [5000, 200000]
location: "India"
collaboration_type: "barter"
tone: "fun"
```

- [ ] **Step 5: Install dependencies**

```bash
cd influencer-match
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 6: Copy .env.example to .env and fill in your keys**

```bash
cp .env.example .env
# Edit .env with your actual OPENAI_API_KEY, SLACK_WEBHOOK_URL, GOOGLE_SHEET_ID
```

- [ ] **Step 7: Commit**

```bash
git init
echo "venv/" >> .gitignore
echo ".env" >> .gitignore
echo "service_account.json" >> .gitignore
git add .
git commit -m "chore: project scaffold — directories, requirements, env template"
```

---

### Task 2: Pydantic Schemas

**Files:**
- Create: `influencer-match/models/schemas.py`

- [ ] **Step 1: Write failing test**

```python
# influencer-match/tests/test_schemas.py
from models.schemas import CampaignBrief, RawProfile, ScoredInfluencer, PipelineState

def test_campaign_brief_defaults():
    brief = CampaignBrief(
        brand_name="FitFuel",
        product_type="protein powder",
        target_audience="gym beginners",
        niches=["fitness"],
    )
    assert brief.collaboration_type == "barter"
    assert brief.follower_range == (5000, 200000)
    assert brief.tone == "fun"

def test_scored_influencer_has_all_fields():
    inf = ScoredInfluencer(
        username="user1",
        followers=10000,
        engagement_rate=4.5,
        niche_score=0.8,
        audience_score=0.6,
        content_quality_score=0.7,
        collaboration_fit_score=1.0,
        final_score=0.74,
    )
    assert inf.why_selected == ""
    assert inf.outreach_message == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd influencer-match
pytest tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.schemas'`

- [ ] **Step 3: Write schemas.py**

```python
# influencer-match/models/schemas.py
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class CampaignBrief(BaseModel):
    brand_name: str
    product_type: str
    target_audience: str
    niches: list[str]
    follower_range: tuple[int, int] = (5000, 200000)
    location: str = "India"
    collaboration_type: str = "barter"  # "barter" | "paid"
    tone: str = "fun"                   # "premium" | "fun" | "educational"


class RawProfile(BaseModel):
    username: str
    full_name: str
    followers: int
    following: int
    post_count: int
    bio: str
    recent_captions: list[str]
    avg_likes: float
    avg_comments: float


class ScoredInfluencer(BaseModel):
    username: str
    followers: int
    engagement_rate: float
    niche_score: float
    audience_score: float
    content_quality_score: float
    collaboration_fit_score: float
    final_score: float
    why_selected: str = ""
    campaign_idea: str = ""
    outreach_message: str = ""
    contact_info: str = ""


class PipelineState(BaseModel):
    campaign: CampaignBrief
    raw_candidates: list[RawProfile] = []
    scored: list[ScoredInfluencer] = []
    top20: list[ScoredInfluencer] = []
    known_usernames: set[str] = set()
    error: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_schemas.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/schemas.py tests/test_schemas.py
git commit -m "feat: Pydantic schemas for campaign brief, profiles, and pipeline state"
```

---

### Task 3: Scrapers — Mock (tests) + Apify (production)

**Files:**
- Create: `influencer-match/scrapers/mock.py` — for unit tests only
- Create: `influencer-match/scrapers/apify_scraper.py` — real production scraper

**Pre-requisite:** Add `APIFY_API_TOKEN=apify_api_xxxx` to your `.env` before running the Apify scraper.
Get your token: [console.apify.com](https://console.apify.com) → Settings → Integrations → API tokens.

**Apify actors used:**
- `apify/instagram-hashtag-scraper` — discovers posts (and author usernames) by hashtag
- `apify/instagram-profile-scraper` — enriches each username with followers, bio, post count, recent posts

**Free tier cost estimate:** ~$0.15/day → ~$4.50/month → fits in $5 free tier.

---

#### 3a: Mock Scraper (for unit tests)

- [ ] **Step 1: Write failing test for mock scraper**

```python
# influencer-match/tests/test_mock_scraper.py
from models.schemas import CampaignBrief
from scrapers.mock import discover_mock_candidates

BRIEF = CampaignBrief(
    brand_name="Perfect Plants",
    product_type="indoor plants",
    target_audience="urban millennials",
    niches=["gardening", "home decor"],
)

def test_discover_returns_target_count():
    candidates = discover_mock_candidates(BRIEF, target=80)
    assert len(candidates) == 80

def test_candidates_respect_follower_range():
    candidates = discover_mock_candidates(BRIEF, target=20)
    for c in candidates:
        lo, hi = BRIEF.follower_range
        assert lo <= c.followers <= hi

def test_candidates_have_niche_keywords():
    candidates = discover_mock_candidates(BRIEF, target=10)
    assert any(
        any(niche in (c.bio + " ".join(c.recent_captions)).lower() for niche in BRIEF.niches)
        for c in candidates
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_mock_scraper.py -v
```

Expected: `ModuleNotFoundError: No module named 'scrapers.mock'`

- [ ] **Step 3: Write scrapers/mock.py**

```python
# influencer-match/scrapers/mock.py
import random
from models.schemas import CampaignBrief, RawProfile

_BASE_USERNAMES = [
    "green_thumb_diaries", "urban_plant_mom", "leafy_living", "plantsofig",
    "the_plant_lady", "succulents_daily", "jungle_vibes", "botanical_bunny",
    "plant_parenthood", "fern_and_friends", "grow_with_grace", "plantaholic",
    "homegarden_hub", "myplantstory", "terrarium_tales", "velvetleaf",
    "rootsandwings", "potted_paradise", "thrive_tribe", "plantandfriend",
    "fitness_fuel_daily", "gym_beginner_life", "healthyhabits_co", "workoutdiary",
    "fitmom_india", "strongstart_fit", "nutritionhub_in", "proteinpacked",
    "decor_obsessed", "aesthetic_home_in", "cozy_corners_ig", "minimalhome",
    "artsy_apartment", "boho_decor_lover", "scandinavian_home", "indianhome",
    "plantdecor_goals", "jungalowstyle", "plantshelfie", "leaflover",
    "urbanjunglefam", "monstera_mania", "cactus_collective", "succulent_squad",
    "gardentherapy", "plantswithpurpose", "indoorjungle", "houseplantsofig",
    "greenvibes_daily", "plantmoodboard", "soilsister", "propagation_station",
    "wateringcan_diaries", "terracotta_tales", "plant_mama_india", "jungalow",
    "plantgang", "pottingmix", "leafandlattice", "growsomething",
    "bloomforever", "rootbound_life", "soilmates", "plantcheck",
    "houseplantnerd", "plantcollector", "thegreenshelf", "vineandfern",
    "claypotlove", "plantdad", "cultivategreen", "mycalathea",
    "alocasialovers", "philodendronphile", "fiddleleaffan", "monsterapeople",
    "snakeplantclub", "pothosgrowth", "peacelilylover", "zzeeplant",
    "stringofpearls", "hoyacommunity", "begoniabeauties", "fernforever",
]

def _make_usernames(count: int) -> list[str]:
    base = _BASE_USERNAMES.copy()
    if count <= len(base):
        return random.sample(base, count)
    extras = [f"{random.choice(base)}_{random.randint(10, 99)}" for _ in range(count - len(base))]
    return base + extras

def _generate_profile(username: str, brief: CampaignBrief) -> RawProfile:
    lo, hi = brief.follower_range
    followers = random.randint(lo, hi)
    avg_likes = followers * random.uniform(0.02, 0.09)
    avg_comments = avg_likes * random.uniform(0.04, 0.15)
    niche_kw = random.choice(brief.niches)
    return RawProfile(
        username=username,
        full_name=username.replace("_", " ").title(),
        followers=followers,
        following=random.randint(100, 2000),
        post_count=random.randint(40, 900),
        bio=f"{niche_kw.capitalize()} enthusiast | {brief.target_audience} | {brief.location} 🌿",
        recent_captions=[
            f"Obsessed with my {brief.product_type} setup! #{niche_kw}lover",
            f"This changed my space completely. #{brief.niches[0]}decor #aesthetic",
            f"Perfect for any {brief.target_audience} 🌿 #{niche_kw}",
        ],
        avg_likes=round(avg_likes, 1),
        avg_comments=round(avg_comments, 1),
    )

def discover_mock_candidates(brief: CampaignBrief, target: int = 80) -> list[RawProfile]:
    return [_generate_profile(u, brief) for u in _make_usernames(target)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mock_scraper.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scrapers/mock.py tests/test_mock_scraper.py
git commit -m "feat: mock scraper for unit tests — generates realistic profiles from campaign brief"
```

---

#### 3b: Apify Scraper (production)

- [ ] **Step 1: Add apify-client to requirements.txt**

```
# add this line to influencer-match/requirements.txt
apify-client==1.7.1
```

```bash
pip install apify-client==1.7.1
```

- [ ] **Step 2: Add APIFY_API_TOKEN to .env.example**

```
# add to influencer-match/.env.example
APIFY_API_TOKEN=apify_api_xxxx
```

Also add it to your `.env` with your real token.

- [ ] **Step 3: Write failing test for Apify scraper**

```python
# influencer-match/tests/test_apify_scraper.py
from unittest.mock import patch, MagicMock
from models.schemas import CampaignBrief
from scrapers.apify_scraper import discover_apify_candidates

BRIEF = CampaignBrief(
    brand_name="Perfect Plants",
    product_type="indoor plants",
    target_audience="urban millennials",
    niches=["gardening", "home decor"],
    follower_range=(5000, 200000),
)

_FAKE_HASHTAG_ITEMS = [
    {"ownerUsername": f"user_{i}", "likesCount": 500, "commentsCount": 30,
     "caption": "Love gardening! #plants"}
    for i in range(60)
]

_FAKE_PROFILE_ITEMS = [
    {
        "username": f"user_{i}",
        "fullName": f"User {i}",
        "followersCount": 20000 + i * 100,
        "followsCount": 500,
        "postsCount": 150,
        "biography": "Gardening and home decor enthusiast",
        "latestPosts": [
            {"likesCount": 400, "commentsCount": 25, "caption": "My plant corner!"},
            {"likesCount": 600, "commentsCount": 40, "caption": "Home decor goals"},
        ],
    }
    for i in range(60)
]


def test_discover_returns_raw_profiles(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")

    with patch("scrapers.apify_scraper.ApifyClient") as MockClient:
        mock_run = MagicMock()
        mock_run.get.return_value = {"status": "SUCCEEDED"}
        MockClient.return_value.actor.return_value.call.return_value = mock_run

        hashtag_dataset = MagicMock()
        hashtag_dataset.iterate_items.return_value = iter(_FAKE_HASHTAG_ITEMS)
        profile_dataset = MagicMock()
        profile_dataset.iterate_items.return_value = iter(_FAKE_PROFILE_ITEMS)

        MockClient.return_value.dataset.return_value.iterate_items.side_effect = [
            iter(_FAKE_HASHTAG_ITEMS),
            iter(_FAKE_PROFILE_ITEMS),
        ]
        MockClient.return_value.actor.return_value.call.return_value.get.return_value = {
            "status": "SUCCEEDED",
            "defaultDatasetId": "fake-dataset-id",
        }

        profiles = discover_apify_candidates(BRIEF, target=20)

    assert len(profiles) >= 1
    assert all(hasattr(p, "username") for p in profiles)
    assert all(hasattr(p, "followers") for p in profiles)


def test_profiles_respect_follower_range(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")

    with patch("scrapers.apify_scraper.ApifyClient") as MockClient:
        MockClient.return_value.actor.return_value.call.return_value.get.return_value = {
            "status": "SUCCEEDED", "defaultDatasetId": "fake-id"
        }
        MockClient.return_value.dataset.return_value.iterate_items.side_effect = [
            iter(_FAKE_HASHTAG_ITEMS),
            iter(_FAKE_PROFILE_ITEMS),
        ]
        profiles = discover_apify_candidates(BRIEF, target=20)

    lo, hi = BRIEF.follower_range
    for p in profiles:
        assert lo <= p.followers <= hi
```

- [ ] **Step 4: Run test to verify it fails**

```bash
pytest tests/test_apify_scraper.py -v
```

Expected: `ModuleNotFoundError: No module named 'scrapers.apify_scraper'`

- [ ] **Step 5: Write scrapers/apify_scraper.py**

```python
# influencer-match/scrapers/apify_scraper.py
import os
from apify_client import ApifyClient
from models.schemas import CampaignBrief, RawProfile

_HASHTAG_ACTOR = "apify/instagram-hashtag-scraper"
_PROFILE_ACTOR = "apify/instagram-profile-scraper"


def _build_hashtags(brief: CampaignBrief) -> list[str]:
    hashtags = list(brief.niches)
    # Add compound hashtags for better coverage
    for niche in brief.niches:
        hashtags.append(niche.replace(" ", "") + "india")
        hashtags.append(niche.replace(" ", "") + "lover")
    return hashtags[:10]  # cap at 10 to control Apify cost


def _run_actor(client: ApifyClient, actor_id: str, run_input: dict) -> list[dict]:
    run = client.actor(actor_id).call(run_input=run_input)
    dataset_id = run.get("defaultDatasetId")
    return list(client.dataset(dataset_id).iterate_items())


def _extract_unique_usernames(hashtag_items: list[dict]) -> list[str]:
    seen = set()
    usernames = []
    for item in hashtag_items:
        username = item.get("ownerUsername", "").strip()
        if username and username not in seen:
            seen.add(username)
            usernames.append(username)
    return usernames


def _profile_to_raw(item: dict, brief: CampaignBrief) -> RawProfile | None:
    username = item.get("username", "")
    followers = item.get("followersCount", 0)
    lo, hi = brief.follower_range
    if not (lo <= followers <= hi):
        return None

    latest = item.get("latestPosts") or []
    if latest:
        avg_likes = sum(p.get("likesCount", 0) for p in latest) / len(latest)
        avg_comments = sum(p.get("commentsCount", 0) for p in latest) / len(latest)
    else:
        avg_likes = 0.0
        avg_comments = 0.0

    captions = [p.get("caption", "") for p in latest[:5] if p.get("caption")]

    return RawProfile(
        username=username,
        full_name=item.get("fullName", username),
        followers=followers,
        following=item.get("followsCount", 0),
        post_count=item.get("postsCount", 0),
        bio=item.get("biography", ""),
        recent_captions=captions,
        avg_likes=round(avg_likes, 1),
        avg_comments=round(avg_comments, 1),
    )


def discover_apify_candidates(brief: CampaignBrief, target: int = 100) -> list[RawProfile]:
    client = ApifyClient(os.environ["APIFY_API_TOKEN"])
    hashtags = _build_hashtags(brief)

    # Step 1: Discover usernames via hashtag scraper
    hashtag_items = _run_actor(client, _HASHTAG_ACTOR, {
        "hashtags": hashtags,
        "resultsLimit": target * 2,  # over-fetch to account for dedup + filter
    })
    usernames = _extract_unique_usernames(hashtag_items)

    if not usernames:
        return []

    # Step 2: Enrich profiles via profile scraper
    profile_items = _run_actor(client, _PROFILE_ACTOR, {
        "usernames": usernames[:target],
    })

    profiles = []
    for item in profile_items:
        profile = _profile_to_raw(item, brief)
        if profile is not None:
            profiles.append(profile)

    return profiles
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_apify_scraper.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 7: Manual smoke test against real Apify (requires real token + internet)**

```python
# run from influencer-match/ with your .env loaded
from dotenv import load_dotenv
load_dotenv()
from models.schemas import CampaignBrief
from scrapers.apify_scraper import discover_apify_candidates

brief = CampaignBrief(
    brand_name="Perfect Plants",
    product_type="indoor plants",
    target_audience="urban millennials",
    niches=["gardening", "homedecor"],
    follower_range=(5000, 200000),
)
profiles = discover_apify_candidates(brief, target=30)
print(f"Got {len(profiles)} profiles")
for p in profiles[:3]:
    print(p.username, p.followers, p.avg_likes)
```

Expected: 10-30 profiles printed with real Instagram usernames.

- [ ] **Step 8: Commit**

```bash
git add scrapers/apify_scraper.py requirements.txt .env.example tests/test_apify_scraper.py
git commit -m "feat: Apify scraper — hashtag discovery + profile enrichment via apify-client"
```

---

### Task 4: Scoring Engine

**Files:**
- Create: `influencer-match/agents/scoring.py`
- Create: `influencer-match/tests/test_scoring.py`

- [ ] **Step 1: Write failing tests**

```python
# influencer-match/tests/test_scoring.py
from models.schemas import CampaignBrief, RawProfile
from agents.scoring import (
    compute_engagement_rate,
    compute_niche_score,
    compute_audience_score,
    compute_content_quality,
    compute_collaboration_fit,
    score_influencer,
    score_candidates,
)

BRIEF = CampaignBrief(
    brand_name="Perfect Plants",
    product_type="indoor plants",
    target_audience="urban millennials",
    niches=["gardening", "home decor"],
    follower_range=(5000, 200000),
    collaboration_type="barter",
    tone="fun",
)

PROFILE = RawProfile(
    username="plant_lover",
    full_name="Plant Lover",
    followers=25000,
    following=500,
    post_count=200,
    bio="Gardening and home decor enthusiast | urban millennials",
    recent_captions=["Love my new gardening setup!", "Home decor goals 🌿"],
    avg_likes=1500.0,
    avg_comments=80.0,
)


def test_engagement_rate_calculation():
    er = compute_engagement_rate(PROFILE)
    # (1500 + 80) / 25000 = 0.0632
    assert abs(er - 0.0632) < 0.001


def test_engagement_rate_zero_followers():
    flat = PROFILE.model_copy(update={"followers": 0})
    assert compute_engagement_rate(flat) == 0.0


def test_niche_score_high_when_bio_matches():
    score = compute_niche_score(PROFILE, BRIEF)
    assert score >= 0.5  # bio contains both "gardening" and "home decor"


def test_niche_score_zero_when_no_match():
    no_match = PROFILE.model_copy(update={"bio": "travel photography", "recent_captions": ["Bali trip!"]})
    score = compute_niche_score(no_match, BRIEF)
    assert score == 0.0


def test_audience_score_matches_target_audience():
    score = compute_audience_score(PROFILE, BRIEF)
    assert score > 0.0  # bio contains "urban" and "millennials"


def test_barter_prefers_small_over_large():
    small = PROFILE.model_copy(update={"followers": 20000})
    large = PROFILE.model_copy(update={"followers": 300000})
    assert compute_collaboration_fit(small, BRIEF) > compute_collaboration_fit(large, BRIEF)


def test_paid_allows_large_creators():
    paid_brief = BRIEF.model_copy(update={"collaboration_type": "paid"})
    large = PROFILE.model_copy(update={"followers": 150000})
    assert compute_collaboration_fit(large, paid_brief) == 1.0


def test_final_score_in_unit_range():
    scored = score_influencer(PROFILE, BRIEF)
    assert 0.0 <= scored.final_score <= 1.0


def test_score_candidates_sorted_descending():
    from scrapers.mock import discover_mock_candidates
    profiles = discover_mock_candidates(BRIEF, target=20)
    scored = score_candidates(profiles, BRIEF)
    scores = [s.final_score for s in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_candidates_returns_all_profiles():
    from scrapers.mock import discover_mock_candidates
    profiles = discover_mock_candidates(BRIEF, target=30)
    scored = score_candidates(profiles, BRIEF)
    assert len(scored) == 30
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scoring.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.scoring'`

- [ ] **Step 3: Write agents/scoring.py**

```python
# influencer-match/agents/scoring.py
from models.schemas import CampaignBrief, RawProfile, ScoredInfluencer


def compute_engagement_rate(profile: RawProfile) -> float:
    if profile.followers == 0:
        return 0.0
    return (profile.avg_likes + profile.avg_comments) / profile.followers


def compute_niche_score(profile: RawProfile, brief: CampaignBrief) -> float:
    text = (profile.bio + " " + " ".join(profile.recent_captions)).lower()
    hits = sum(1 for niche in brief.niches if niche.lower() in text)
    return min(hits / max(len(brief.niches), 1), 1.0)


def compute_audience_score(profile: RawProfile, brief: CampaignBrief) -> float:
    text = (profile.bio + " " + " ".join(profile.recent_captions)).lower()
    words = brief.target_audience.lower().split()
    hits = sum(1 for w in words if w in text)
    return min(hits / max(len(words), 1), 1.0)


def compute_content_quality(profile: RawProfile) -> float:
    post_score = min(profile.post_count / 500, 1.0)
    avg_len = sum(len(c) for c in profile.recent_captions) / max(len(profile.recent_captions), 1)
    caption_score = min(avg_len / 150, 1.0)
    return 0.5 * post_score + 0.5 * caption_score


def compute_collaboration_fit(profile: RawProfile, brief: CampaignBrief) -> float:
    if brief.collaboration_type == "barter":
        if 5_000 <= profile.followers <= 50_000:
            return 1.0
        if profile.followers < 5_000:
            return 0.4
        return max(0.0, 1.0 - (profile.followers - 50_000) / 450_000)
    lo, hi = brief.follower_range
    if lo <= profile.followers <= hi:
        return 1.0
    return 0.5


def score_influencer(profile: RawProfile, brief: CampaignBrief) -> ScoredInfluencer:
    er = compute_engagement_rate(profile)
    eng_score = min(er / 0.06, 1.0)  # 6% ER is benchmark excellent
    niche = compute_niche_score(profile, brief)
    audience = compute_audience_score(profile, brief)
    content = compute_content_quality(profile)
    collab = compute_collaboration_fit(profile, brief)

    final = (
        0.30 * eng_score
        + 0.25 * niche
        + 0.20 * audience
        + 0.15 * content
        + 0.10 * collab
    )

    return ScoredInfluencer(
        username=profile.username,
        followers=profile.followers,
        engagement_rate=round(er * 100, 2),
        niche_score=round(niche, 3),
        audience_score=round(audience, 3),
        content_quality_score=round(content, 3),
        collaboration_fit_score=round(collab, 3),
        final_score=round(final, 3),
    )


def score_candidates(profiles: list[RawProfile], brief: CampaignBrief) -> list[ScoredInfluencer]:
    return sorted(
        [score_influencer(p, brief) for p in profiles],
        key=lambda x: x.final_score,
        reverse=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scoring.py -v
```

Expected: 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/scoring.py tests/test_scoring.py
git commit -m "feat: dynamic scoring engine with 5 weighted sub-scores"
```

---

### Task 5: Google Sheets Output + Dedup

**Files:**
- Create: `influencer-match/output/sheets.py`
- Create: `influencer-match/tests/test_dedup.py`

**Before starting:** You need a Google Cloud service account with Google Sheets API enabled. Download the JSON key, save it to `influencer-match/service_account.json`. Share your target Google Sheet with the service account email.

- [ ] **Step 1: Write failing test**

```python
# influencer-match/tests/test_dedup.py
from unittest.mock import patch, MagicMock
from output.sheets import get_existing_usernames


def test_returns_usernames_from_all_tab():
    mock_records = [{"username": "user1"}, {"username": "user2"}, {"username": "user3"}]
    with patch("output.sheets.get_sheet_client") as mock_client:
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = mock_records
        mock_client.return_value.open_by_key.return_value.worksheet.return_value = mock_ws

        result = get_existing_usernames("fake-sheet-id")
        assert result == {"user1", "user2", "user3"}


def test_returns_empty_set_when_all_tab_missing():
    import gspread
    with patch("output.sheets.get_sheet_client") as mock_client:
        mock_client.return_value.open_by_key.return_value.worksheet.side_effect = (
            gspread.exceptions.WorksheetNotFound
        )
        result = get_existing_usernames("fake-sheet-id")
        assert result == set()


def test_returns_empty_set_when_records_have_no_username_key():
    mock_records = [{"name": "user1"}, {}]
    with patch("output.sheets.get_sheet_client") as mock_client:
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = mock_records
        mock_client.return_value.open_by_key.return_value.worksheet.return_value = mock_ws

        result = get_existing_usernames("fake-sheet-id")
        assert result == set()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_dedup.py -v
```

Expected: `ModuleNotFoundError: No module named 'output.sheets'`

- [ ] **Step 3: Write output/sheets.py**

```python
# influencer-match/output/sheets.py
import os
from datetime import date
import gspread
from google.oauth2.service_account import Credentials
from models.schemas import ScoredInfluencer

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_HEADERS = [
    "username", "followers", "engagement_rate", "final_score",
    "why_selected", "campaign_idea", "outreach_message", "contact_info",
]


def get_sheet_client() -> gspread.Client:
    creds = Credentials.from_service_account_file("service_account.json", scopes=_SCOPES)
    return gspread.authorize(creds)


def get_existing_usernames(sheet_id: str) -> set[str]:
    client = get_sheet_client()
    try:
        ws = client.open_by_key(sheet_id).worksheet("All")
    except gspread.exceptions.WorksheetNotFound:
        return set()
    records = ws.get_all_records()
    return {r["username"] for r in records if "username" in r}


def write_influencers(influencers: list[ScoredInfluencer], sheet_id: str) -> int:
    client = get_sheet_client()
    sh = client.open_by_key(sheet_id)
    today = str(date.today())

    try:
        ws = sh.worksheet(today)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=today, rows=25, cols=len(_HEADERS))
        ws.append_row(_HEADERS)

    rows = [
        [
            i.username, i.followers, i.engagement_rate, i.final_score,
            i.why_selected, i.campaign_idea, i.outreach_message, i.contact_info,
        ]
        for i in influencers
    ]
    ws.append_rows(rows)

    try:
        all_ws = sh.worksheet("All")
    except gspread.exceptions.WorksheetNotFound:
        all_ws = sh.add_worksheet(title="All", rows=1000, cols=3)
        all_ws.append_row(["username", "date_added", "final_score"])

    all_ws.append_rows([[i.username, today, i.final_score] for i in influencers])
    return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_dedup.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add output/sheets.py tests/test_dedup.py
git commit -m "feat: Google Sheets writer with daily tab creation and dedup via All tab"
```

---

### Task 6: Slack Output

**Files:**
- Create: `influencer-match/output/slack.py`

- [ ] **Step 1: Write failing test**

```python
# influencer-match/tests/test_slack.py
from unittest.mock import patch, MagicMock
from models.schemas import CampaignBrief, ScoredInfluencer
from output.slack import post_daily_summary

BRIEF = CampaignBrief(
    brand_name="Perfect Plants",
    product_type="indoor plants",
    target_audience="urban millennials",
    niches=["gardening"],
)

def _make_influencer(username: str, score: float) -> ScoredInfluencer:
    return ScoredInfluencer(
        username=username, followers=20000, engagement_rate=4.5,
        niche_score=0.8, audience_score=0.6, content_quality_score=0.7,
        collaboration_fit_score=1.0, final_score=score,
        campaign_idea="Unboxing reel", outreach_message="Hi there!",
    )

def test_posts_to_slack_webhook(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/fake")
    influencers = [_make_influencer(f"user{i}", 0.9 - i * 0.05) for i in range(20)]

    with patch("output.slack.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        post_daily_summary(influencers, BRIEF)
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert "Perfect Plants" in payload["text"]
        assert "user0" in payload["text"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_slack.py -v
```

Expected: `ModuleNotFoundError: No module named 'output.slack'`

- [ ] **Step 3: Write output/slack.py**

```python
# influencer-match/output/slack.py
import os
import httpx
from models.schemas import ScoredInfluencer, CampaignBrief


def post_daily_summary(influencers: list[ScoredInfluencer], brief: CampaignBrief) -> None:
    webhook_url = os.environ["SLACK_WEBHOOK_URL"]
    top5 = influencers[:5]
    lines = [
        f"*InfluencerMatch Daily Report — {brief.brand_name}*",
        f"Found {len(influencers)} candidates today. Top 5:\n",
    ]
    for i, inf in enumerate(top5, 1):
        lines.append(
            f"{i}. @{inf.username} — {inf.followers:,} followers | "
            f"{inf.engagement_rate}% ER | Score: {inf.final_score}"
        )
    lines.append("\nFull list in Google Sheets ✅")
    httpx.post(webhook_url, json={"text": "\n".join(lines)})
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_slack.py -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add output/slack.py tests/test_slack.py
git commit -m "feat: Slack webhook posts daily top-5 summary with brand name and scores"
```

---

### Task 7: Intelligence Agent (GPT-4o)

**Files:**
- Create: `influencer-match/agents/intelligence.py`

- [ ] **Step 1: Write failing test**

```python
# influencer-match/tests/test_intelligence.py
from unittest.mock import patch, MagicMock
from models.schemas import CampaignBrief, ScoredInfluencer
from agents.intelligence import enrich_with_intelligence

BRIEF = CampaignBrief(
    brand_name="Perfect Plants",
    product_type="indoor plants",
    target_audience="urban millennials",
    niches=["gardening", "home decor"],
    tone="fun",
    collaboration_type="barter",
)

INF = ScoredInfluencer(
    username="plant_lover",
    followers=25000,
    engagement_rate=6.3,
    niche_score=0.8,
    audience_score=0.7,
    content_quality_score=0.6,
    collaboration_fit_score=1.0,
    final_score=0.76,
)

def test_enrichment_fills_why_and_idea_and_outreach(monkeypatch):
    import json
    fake_idea_response = MagicMock()
    fake_idea_response.choices[0].message.content = json.dumps({
        "why": "Great engagement in the gardening niche",
        "campaign_idea": "60-second plant care tutorial reel",
    })
    fake_outreach_response = MagicMock()
    fake_outreach_response.choices[0].message.content = (
        "Hey @plant_lover! Love your plant content. "
        "We'd love to send you some of our indoor plants as a barter collab. Interested?"
    )

    mock_completions = MagicMock()
    mock_completions.create.side_effect = [fake_idea_response, fake_outreach_response]

    with patch("agents.intelligence.client") as mock_client:
        mock_client.chat.completions = mock_completions
        result = enrich_with_intelligence(INF.model_copy(), BRIEF)

    assert result.why_selected == "Great engagement in the gardening niche"
    assert result.campaign_idea == "60-second plant care tutorial reel"
    assert "@plant_lover" in result.outreach_message


def test_enrichment_does_not_mutate_original():
    original = INF.model_copy()
    import json
    fake_idea_response = MagicMock()
    fake_idea_response.choices[0].message.content = json.dumps({
        "why": "reason", "campaign_idea": "idea"
    })
    fake_outreach_response = MagicMock()
    fake_outreach_response.choices[0].message.content = "outreach text"
    mock_completions = MagicMock()
    mock_completions.create.side_effect = [fake_idea_response, fake_outreach_response]

    with patch("agents.intelligence.client") as mock_client:
        mock_client.chat.completions = mock_completions
        enrich_with_intelligence(INF.model_copy(), BRIEF)

    assert INF.why_selected == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_intelligence.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.intelligence'`

- [ ] **Step 3: Write agents/intelligence.py**

```python
# influencer-match/agents/intelligence.py
import json
import os
from openai import OpenAI
from models.schemas import ScoredInfluencer, CampaignBrief

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

_IDEA_PROMPT = """\
You are a campaign strategist. Given a campaign brief and an influencer profile, produce:
1. why_selected — one sentence explaining why this influencer fits the campaign
2. campaign_idea — one specific content idea (format + topic)

Brief: {brief}
Influencer: @{username}, {followers:,} followers, {engagement_rate}% engagement rate

Respond as JSON only: {{"why": "...", "campaign_idea": "..."}}"""

_OUTREACH_PROMPT = """\
Write a short, {tone} outreach DM for a brand collaboration.
- Brand: {brand_name} ({product_type})
- Influencer: @{username}
- Campaign idea: {campaign_idea}
- Collaboration type: {collaboration_type}

Rules: 3 sentences max. Mention their content specifically. State the collab type clearly. No generic openers."""


def enrich_with_intelligence(influencer: ScoredInfluencer, brief: CampaignBrief) -> ScoredInfluencer:
    brief_str = (
        f"{brief.brand_name} sells {brief.product_type} for {brief.target_audience}. "
        f"Niches: {', '.join(brief.niches)}."
    )

    idea_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": _IDEA_PROMPT.format(
            brief=brief_str,
            username=influencer.username,
            followers=influencer.followers,
            engagement_rate=influencer.engagement_rate,
        )}],
        response_format={"type": "json_object"},
    )
    idea_data = json.loads(idea_resp.choices[0].message.content)

    outreach_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": _OUTREACH_PROMPT.format(
            tone=brief.tone,
            brand_name=brief.brand_name,
            product_type=brief.product_type,
            username=influencer.username,
            campaign_idea=idea_data.get("campaign_idea", "brand collaboration"),
            collaboration_type=brief.collaboration_type,
        )}],
    )

    influencer.why_selected = idea_data.get("why", "")
    influencer.campaign_idea = idea_data.get("campaign_idea", "")
    influencer.outreach_message = outreach_resp.choices[0].message.content
    return influencer
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_intelligence.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/intelligence.py tests/test_intelligence.py
git commit -m "feat: GPT-4o intelligence agent generates campaign ideas and outreach messages"
```

---

### Task 8: LangGraph Pipeline Graph

**Files:**
- Create: `influencer-match/agents/graph.py`
- Create: `influencer-match/tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# influencer-match/tests/test_pipeline.py
import pytest
from unittest.mock import patch
from models.schemas import CampaignBrief, ScoredInfluencer
from agents.graph import build_graph, PipelineState

BRIEF = CampaignBrief(
    brand_name="Perfect Plants",
    product_type="indoor plants",
    target_audience="urban millennials",
    niches=["gardening", "home decor"],
    follower_range=(5000, 200000),
    collaboration_type="barter",
    tone="fun",
)


def _mock_enrich(influencer: ScoredInfluencer, brief: CampaignBrief) -> ScoredInfluencer:
    influencer.why_selected = "Mock reason"
    influencer.campaign_idea = "Mock reel idea"
    influencer.outreach_message = "Mock outreach DM"
    return influencer


def test_pipeline_returns_20_influencers(monkeypatch):
    monkeypatch.setenv("USE_MOCK_SCRAPER", "true")
    import agents.intelligence as intel_mod
    monkeypatch.setattr(intel_mod, "enrich_with_intelligence", _mock_enrich)

    graph = build_graph()
    initial: PipelineState = {
        "campaign": BRIEF,
        "raw_candidates": [],
        "scored": [],
        "top20": [],
        "known_usernames": set(),
        "error": None,
    }
    result = graph.invoke(initial)
    assert len(result["top20"]) == 20


def test_pipeline_deduplicates_known_usernames(monkeypatch):
    monkeypatch.setenv("USE_MOCK_SCRAPER", "true")
    import agents.intelligence as intel_mod
    monkeypatch.setattr(intel_mod, "enrich_with_intelligence", _mock_enrich)

    # Pre-populate known_usernames with names that will appear in mock data
    from scrapers.mock import discover_mock_candidates
    preview = discover_mock_candidates(BRIEF, target=30)
    known = {p.username for p in preview[:15]}

    graph = build_graph()
    result = graph.invoke({
        "campaign": BRIEF,
        "raw_candidates": [],
        "scored": [],
        "top20": [],
        "known_usernames": known,
        "error": None,
    })
    returned_usernames = {inf.username for inf in result["top20"]}
    assert returned_usernames.isdisjoint(known)


def test_pipeline_top20_sorted_by_score(monkeypatch):
    monkeypatch.setenv("USE_MOCK_SCRAPER", "true")
    import agents.intelligence as intel_mod
    monkeypatch.setattr(intel_mod, "enrich_with_intelligence", _mock_enrich)

    graph = build_graph()
    result = graph.invoke({
        "campaign": BRIEF,
        "raw_candidates": [],
        "scored": [],
        "top20": [],
        "known_usernames": set(),
        "error": None,
    })
    scores = [inf.final_score for inf in result["top20"]]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.graph'`

- [ ] **Step 3: Write agents/graph.py**

```python
# influencer-match/agents/graph.py
import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from models.schemas import CampaignBrief, RawProfile, ScoredInfluencer
from agents.scoring import score_candidates
from agents.intelligence import enrich_with_intelligence


class PipelineState(TypedDict):
    campaign: CampaignBrief
    raw_candidates: list[RawProfile]
    scored: list[ScoredInfluencer]
    top20: list[ScoredInfluencer]
    known_usernames: set[str]
    error: Optional[str]


def _get_scraper():
    # Use mock scraper in tests (set USE_MOCK_SCRAPER=true in env), Apify in production
    if os.environ.get("USE_MOCK_SCRAPER", "false").lower() == "true":
        from scrapers.mock import discover_mock_candidates
        return discover_mock_candidates
    from scrapers.apify_scraper import discover_apify_candidates
    return discover_apify_candidates


def _discovery_node(state: PipelineState) -> PipelineState:
    scraper = _get_scraper()
    candidates = scraper(state["campaign"], target=100)
    filtered = [c for c in candidates if c.username not in state["known_usernames"]]
    state["raw_candidates"] = filtered
    return state


def _scoring_node(state: PipelineState) -> PipelineState:
    scored = score_candidates(state["raw_candidates"], state["campaign"])
    state["scored"] = scored
    state["top20"] = scored[:20]
    return state


def _intelligence_node(state: PipelineState) -> PipelineState:
    enriched = [
        enrich_with_intelligence(inf, state["campaign"])
        for inf in state["top20"]
    ]
    state["top20"] = enriched
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("discover", _discovery_node)
    graph.add_node("score", _scoring_node)
    graph.add_node("intelligence", _intelligence_node)

    graph.set_entry_point("discover")
    graph.add_edge("discover", "score")
    graph.add_edge("score", "intelligence")
    graph.add_edge("intelligence", END)

    return graph.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/graph.py tests/test_pipeline.py
git commit -m "feat: LangGraph pipeline — discover, score, intelligence nodes wired end-to-end"
```

---

### Task 9: FastAPI Endpoint

**Files:**
- Create: `influencer-match/main.py`

- [ ] **Step 1: Write main.py**

```python
# influencer-match/main.py
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from models.schemas import CampaignBrief, ScoredInfluencer
from agents.graph import build_graph, PipelineState
from output.sheets import get_existing_usernames, write_influencers
from output.slack import post_daily_summary

load_dotenv()
app = FastAPI(title="InfluencerMatch AI", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_pipeline():
    config_path = Path("config/campaign.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="config/campaign.yaml not found")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    campaign = CampaignBrief(**data)
    sheet_id = os.environ["GOOGLE_SHEET_ID"]

    known = get_existing_usernames(sheet_id)

    graph = build_graph()
    initial: PipelineState = {
        "campaign": campaign,
        "raw_candidates": [],
        "scored": [],
        "top20": [],
        "known_usernames": known,
        "error": None,
    }
    result = graph.invoke(initial)

    top20: list[ScoredInfluencer] = result["top20"]
    if not top20:
        raise HTTPException(status_code=500, detail="Pipeline produced 0 candidates")

    written = write_influencers(top20, sheet_id)
    post_daily_summary(top20, campaign)

    return {
        "status": "success",
        "candidates_written": written,
        "brand": campaign.brand_name,
    }
```

- [ ] **Step 2: Start the server and verify health endpoint**

```bash
cd influencer-match
uvicorn main:app --reload --port 8001
```

In another terminal:
```bash
curl http://localhost:8001/health
```

Expected:
```json
{"status": "ok"}
```

- [ ] **Step 3: Test /run with mock data (Google Sheets + Slack must be configured)**

```bash
curl -X POST http://localhost:8001/run
```

Expected:
```json
{"status": "success", "candidates_written": 20, "brand": "Perfect Plants"}
```

Check your Google Sheet — new tab named today's date should appear with 20 rows.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: FastAPI /run endpoint — reads campaign.yaml, runs pipeline, writes to Sheets + Slack"
```

---

### Task 10: n8n Workflow

**Files:** No code files — configured in n8n UI.

- [ ] **Step 1: In n8n, create a new workflow named "InfluencerMatch Daily"**

- [ ] **Step 2: Add a Schedule trigger node**

Set to run daily at 8:00 AM (or your preferred time).

- [ ] **Step 3: Add an HTTP Request node**

- Method: `POST`
- URL: `http://localhost:8001/run` (or your deployed URL if hosted)
- Response Format: `JSON`

- [ ] **Step 4: Add an IF node to check success**

Condition: `{{ $json.status }}` equals `"success"`

- [ ] **Step 5: (True branch) Add a Slack node or log node**

Send a message: `InfluencerMatch ran — {{ $json.candidates_written }} candidates written for {{ $json.brand }}.`

- [ ] **Step 6: (False branch) Add error notification**

Send Slack alert: `InfluencerMatch pipeline failed. Check the server logs.`

- [ ] **Step 7: Test the workflow manually in n8n**

Click "Test Workflow" — verify it hits `/run`, gets a success response, and posts the Slack message.

---

## Self-Review

### Spec Coverage

| PRD Requirement | Task |
|---|---|
| Campaign input layer (brand, niches, follower range, etc.) | Task 2 — CampaignBrief schema + campaign.yaml |
| Hashtag-based discovery | Task 3b — apify_scraper.py via `apify/instagram-hashtag-scraper` |
| Engagement + keyword enrichment | Task 3 (mock), Task 4 (scoring) |
| Dynamic scoring (engagement, niche, audience, content, collab) | Task 4 — scoring.py with 5 weighted sub-scores |
| LLM for campaign ideas + outreach | Task 7 — intelligence.py (GPT-4o) |
| Google Sheets output | Task 5 — sheets.py |
| Slack output | Task 6 — slack.py |
| Deduplication across days | Task 5 — dedup via "All" tab |
| 20 unique influencers/day | Tasks 3+4+8 — discover 100, filter dedup, score, take top 20 |
| n8n orchestration | Task 10 |
| Brand-agnostic (any niche/brand) | Tasks 2+3+4+7 — all driven by CampaignBrief config |

### Gaps Identified — None. All PRD requirements are covered.

### Placeholder Scan — None. Every task contains actual code, exact commands, and expected output.

### Type Consistency
- `PipelineState` TypedDict uses `CampaignBrief`, `RawProfile`, `ScoredInfluencer` — same Pydantic models throughout.
- `score_candidates` in Task 4 returns `list[ScoredInfluencer]` — matches what `_scoring_node` in Task 8 assigns to `state["top20"]`.
- `enrich_with_intelligence(influencer, brief)` signature in Task 7 matches call in Task 8.
- `write_influencers(top20, sheet_id)` signature in Task 5 matches call in Task 9.
