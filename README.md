### AI Employee Dashboard Agent – Technical & Product Spec  
Application Link: [https://dashboard-agent-799818976326.us-central1.run.app/](url)

#### 1. Current Stack & Infrastructure
| Layer              | Technology / Choice                                 | Notes |
|--------------------|-----------------------------------------------------|-------|
| Backend            | FastAPI (Python 3.10+)                              | Lightweight, async-ready |
| AI Query Parser    | Google Vertex AI → `gemini-1.5-pro-001` (optional) <br>Fallback: rule-based keyword parser | Works even if Vertex AI keys missing |
| Data Layer         | In-memory Pandas DataFrames (cached)                | Fake data generated on first call |
| Frontend           | Pure HTML + CSS + vanilla JS (no React/Vue)        | Single-page, Claude-style minimal UI |
| Charts             | Plotly.js 2.27 (via CDN)                            | Responsive, good-looking |
| Deployment ready   | Any platform that runs Python + uvicorn/gunicorn    | Docker-ready with minimal changes |
| Hosting examples   | Cloud Run, Render, Fly.io, Railway, Vercel (with serverless adapter), GCP/AWS | Zero-ops possible |

#### 2. Core Logic Flow
```
User query → POST /generate-dashboard
   ↓
parse_query_with_ai() → tries Gemini → falls back to keyword parser
   ↓
filter_data() → applies simple column filters + time period (quarter/month)
   ↓
generate_dashboard_html() → 
      • picks dashboard_type (attrition | hours | band | etc.)
      • builds 3–4 Plotly figures + smart KPI cards
      • returns raw HTML string
   ↓
Frontend replaces #dashboardContainer with the HTML
```

#### 3. Current Capabilities (what actually works today)
- Natural-language → dashboard in <3 sec (even without Gemini)
- 7 pre-defined dashboard types with professional visuals
- Time-period filtering (this quarter / this month / last 90 days)
- Fully client-side rendering → zero latency after response

#### 4. Critical Parts to Improve for Real HR Use

| Area                     | Current State                            | Production-Grade Target |
|--------------------------|------------------------------------------|-------------------------|
| Data source              | Hard-coded fake data                     | Secure connection to Workday / BambooHR / SuccessFactors / SAP via API or export |
| Authentication           | None                                     | OAuth2 + role-based access (HRBP, Manager, Exec) |
| Query parsing accuracy   | 60–70 % with Gemini, 40 % fallback      | Fine-tune smaller model (Gemini Flash / Llama-3 8B) on 500+ real HR queries |
| Filters & granularity    | Only basic column equals                 | Multi-condition, date ranges, regex, top-N, exclusions |
| Caching                  | In-memory only (dies on restart)         | Redis + per-user cache + data refresh scheduler |
| Export                   | None                                     | PDF / PNG / CSV export buttons |
| Responsiveness           | Works on mobile but not optimized        | Proper mobile layout + touch-friendly |
| Accessibility            | Minimal                                  | WCAG 2.1 AA |
| Audit & compliance       | None                                     | Log every query + generated dashboard (GDPR/CCPA) |
| Multi-language           | English only                             | Spanish / French / German prompts & UI |
| Real-time data           | Static fake data                         | WebSocket or periodic polling for live updates |

#### 5. Realistic Roadmap to Make It HR-Team Useful (MVP → v1.0)

| Phase | Duration | Deliverable | Value to HR |
|-------|----------|-------------|-------------|
| 0 (current) | – | Prototype with fake data | Demo & stakeholder buy-in |
| 1 | 2–3 weeks | Connect to real HRIS (read-only API or nightly CSV) + basic auth | Real numbers, no more “fake” objection |
| 2 | 3–4 weeks | Fine-tune query parser on real past HR questions + add 5 more dashboard types | 90 %+ intent recognition |
| 3 | 2 weeks | Role-based views + export + audit log | Compliance-ready |
| 4 | 2 weeks | Mobile optimization + Slack/Teams bot integration | Used daily by managers |
| 5 | ongoing | Scheduled reports + anomaly alerts | Proactive people analytics |

#### 6. Why This Has High Chance of Adoption in HR Teams
- Zero learning curve: HR asks questions in plain English
- Instant professional dashboards (better than most BI tools for ad-hoc)
- No need to know Power BI / Tableau / Workday Prism
- Managers love it because they get answers in 5 seconds instead of opening a ticket
- HRBPs use it in leadership meetings → looks cutting-edge with almost zero cost

#### Bottom Line
~2 months of focused engineering (data connector + auth + better parser)
