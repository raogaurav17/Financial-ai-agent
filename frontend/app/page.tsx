"use client";

import { FormEvent, useMemo, useState } from "react";

type Result = Record<string, unknown>;
type Message = { role: "user" | "assistant"; content: string; result?: Result };

const examples = [
  { label: "Latest quote", query: "fetch market data for AAPL" },
  { label: "Risk snapshot", query: "assess risk for TSLA" },
  { label: "30-day trend", query: "predict trends for GOOGL" },
  { label: "Explain a market move", query: "Explain why NVDA has been volatile" },
];

function formatValue(value: unknown): string {
  if (typeof value === "number") return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return String(value);
}

function ResultCard({ result }: { result: Result }) {
  const entries = Object.entries(result);
  return (
    <div className="result-card">
      {entries.map(([key, value]) => (
        <div className="result-row" key={key}>
          <span>{key.replaceAll("_", " ")}</span>
          <strong>{formatValue(value)}</strong>
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [apiKey, setApiKey] = useState("my-secret-standard-key");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState("Ready for analysis");
  const apiUrl = useMemo(() => process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000", []);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setStatusText("Running finance service…");
    setMessages((current) => [...current, { role: "user", content: trimmed }]);
    setQuery("");

    try {
      const response = await fetch(`${apiUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-api-key": apiKey },
        body: JSON.stringify({ query: trimmed }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "The request could not be completed.");
      setMessages((current) => [
        ...current,
        { role: "assistant", content: payload.response, result: payload.result },
      ]);
      setStatusText(payload.result ? "Direct result returned · no LLM used" : "AI response returned");
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: error instanceof Error ? error.message : "Something went wrong." },
      ]);
      setStatusText("Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">F</div>
          <div><strong>financial ai agent</strong><small>financial intelligence</small></div>
        </div>
        <div className="nav-label">WORKSPACE</div>
        <button className="nav-item active"><span>⌂</span> Overview</button>
        <button className="nav-item"><span>◷</span> Market watch</button>
        <button className="nav-item"><span>⌁</span> Risk lab</button>
        <button className="nav-item"><span>◈</span> Portfolios</button>
        <div className="sidebar-bottom">
          <div className="service-card"><span className="pulse" /> API service online<small>FastAPI · port 8000</small></div>
          <div className="profile"><div className="avatar">GA</div><div><strong>Guest analyst</strong><small>Standard workspace</small></div><span>•••</span></div>
        </div>
      </aside>

      <section className="content">
        <header className="topbar">
          <div><span className="eyebrow">THURSDAY, SEPTEMBER 24, 2026</span><h1>Good afternoon, Gaurav <span>✦</span></h1></div>
          <div className="top-actions"><button className="icon-button" aria-label="Notifications">♢</button><button className="outline-button">Documentation ↗</button></div>
        </header>

        <div className="hero-grid">
          <section className="hero">
            <div className="hero-kicker"><span className="spark">✦</span> FINANCE COPILOT</div>
            <h2>What would you like<br />to <em>understand?</em></h2>
            <p>Ask about markets, risk, or your portfolio. Deterministic requests go straight to the finance service for fast, transparent results.</p>
            <div className="example-list">{examples.map((example) => <button key={example.label} onClick={() => setQuery(example.query)}>{example.label}<span>↗</span></button>)}</div>
          </section>
          <section className="metric-panel">
            <div className="panel-heading"><span>MARKET PULSE</span><span className="live"><i /> LIVE</span></div>
            <div className="pulse-chart"><div className="chart-line" /><span className="chart-dot dot-one" /><span className="chart-dot dot-two" /></div>
            <div className="pulse-footer"><div><small>S&P 500</small><strong>5,722.26</strong></div><div className="positive">+0.41%<small>today</small></div></div>
            <div className="market-strip"><span>NASDAQ <b>+0.68%</b></span><span>DOW <b>+0.18%</b></span></div>
          </section>
        </div>

        <section className="query-section">
          <div className="section-heading"><div><span className="eyebrow">COMMAND CENTER</span><h3>Ask the market anything</h3></div><span className={`status ${loading ? "working" : ""}`}><i /> {statusText}</span></div>
          <form className="query-box" onSubmit={submit}>
            <span className="query-icon">⌕</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. assess risk for TSLA, or compare AAPL and MSFT…" />
            <button type="submit" disabled={loading || !query.trim()}>{loading ? "Working…" : "Run analysis"} <span>↗</span></button>
          </form>
          <div className="query-meta"><span>↳ Direct service routing enabled</span><label>API key <input value={apiKey} onChange={(event) => setApiKey(event.target.value)} type="password" /></label></div>
        </section>

        <section className="conversation">
          {messages.length === 0 ? <div className="empty-state"><div className="empty-icon">✦</div><h3>Your analysis will appear here</h3><p>Start with a quote, risk snapshot, or ask the copilot to explain something.</p></div> : messages.map((message, index) => (
            <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="message-label">{message.role === "user" ? "YOU" : "FINANCIAL AI AGENT"} <span>{message.role === "assistant" && "✦"}</span></div>
              <p>{message.content}</p>
              {message.result && <ResultCard result={message.result} />}
            </div>
          ))}
        </section>
      </section>
    </main>
  );
}
