import React, { useState } from "react";
import { useMutation } from "@apollo/client";
import { SCORE_CAMPAIGN } from "../graphql/queries";

type ModelMode = "XGB" | "ENSEMBLE";

const SAMPLE_CAMPAIGN = `Emergency medical fundraiser

Please help cover urgent treatment and travel costs. Donations will go toward hospital bills, medication, and temporary housing for the family.`;

function percent(value?: number | null) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "N/A";
}

export function CampaignScorer() {
  const [campaignText, setCampaignText] = useState(SAMPLE_CAMPAIGN);
  const [modelMode, setModelMode] = useState<ModelMode>("ENSEMBLE");
  const [scoreCampaign, { data, loading, error }] = useMutation(SCORE_CAMPAIGN);

  const result = data?.scoreCampaign;
  const labelColor = result?.label === 1 ? "#dc2626" : "#16a34a";
  const outputText = result
    ? `${result.labelText} risk - ${percent(result.fraudScore)}`
    : "Score will appear here";

  const handleSubmit = async () => {
    const text = campaignText.trim();
    if (!text) return;
    await scoreCampaign({
      variables: {
        campaignText: text,
        modelMode,
      },
    });
  };

  return (
    <div style={{ maxWidth: 780, margin: "0 auto", padding: "32px 24px" }}>
      <section style={{
        background: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: 24,
        boxShadow: "0 1px 3px rgba(15, 23, 42, 0.06)",
      }}>
        <h1 style={{ fontSize: 24, lineHeight: 1.2, fontWeight: 750, margin: "0 0 18px" }}>
          Fundraiser Campaign
        </h1>

        <textarea
          value={campaignText}
          onChange={event => setCampaignText(event.target.value)}
          rows={9}
          style={{
            width: "100%",
            boxSizing: "border-box",
            resize: "vertical",
            border: "1px solid #cbd5e1",
            borderRadius: 8,
            padding: 14,
            fontSize: 14,
            lineHeight: 1.5,
            color: "#0f172a",
            background: "#f8fafc",
            outlineColor: "#2563eb",
          }}
        />

        <div style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 14,
          marginTop: 16,
        }}>
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            border: "1px solid #cbd5e1",
            borderRadius: 8,
            overflow: "hidden",
            minWidth: 280,
          }}>
            {[
              { key: "XGB", label: "Only XGBoost" },
              { key: "ENSEMBLE", label: "XGBoost + LoRA" },
            ].map(option => {
              const selected = modelMode === option.key;
              return (
                <button
                  key={option.key}
                  type="button"
                  onClick={() => setModelMode(option.key as ModelMode)}
                  style={{
                    border: "none",
                    borderRight: option.key === "XGB" ? "1px solid #cbd5e1" : "none",
                    background: selected ? "#1d4ed8" : "#fff",
                    color: selected ? "#fff" : "#334155",
                    padding: "10px 12px",
                    fontSize: 13,
                    fontWeight: 700,
                    cursor: "pointer",
                  }}
                >
                  {option.label}
                </button>
              );
            })}
          </div>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={loading || !campaignText.trim()}
            style={{
              minWidth: 190,
              border: "none",
              borderRadius: 8,
              background: loading || !campaignText.trim() ? "#94a3b8" : "#0f172a",
              color: "#fff",
              padding: "11px 18px",
              fontSize: 14,
              fontWeight: 750,
              cursor: loading || !campaignText.trim() ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Scoring..." : "Get Fraud risk score"}
          </button>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 1fr)",
          gap: 12,
          marginTop: 22,
        }}>
          <div style={{
            border: "1px solid #cbd5e1",
            borderRadius: 8,
            background: "#f8fafc",
            padding: 16,
          }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#64748b", marginBottom: 8 }}>
              Output
            </div>
            <div style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "baseline",
              gap: 10,
            }}>
              <span style={{
                color: result ? labelColor : "#64748b",
                fontSize: result ? 30 : 18,
                lineHeight: 1.1,
                fontWeight: 800,
              }}>
                {outputText}
              </span>
              {result && (
                <span style={{ color: "#64748b", fontSize: 13 }}>
                  threshold {percent(result.threshold)} · mode {result.modelMode === "ENSEMBLE" ? "XGBoost + LoRA" : "Only XGBoost"}
                </span>
              )}
            </div>
            {result && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 12, fontSize: 12 }}>
                <span style={{ color: "#334155" }}>XGBoost: {percent(result.xgbScore)}</span>
                <span style={{ color: "#334155" }}>LoRA: {percent(result.loraScore)}</span>
                <span style={{ color: "#334155" }}>LoRA weight: {percent(result.loraWeight)}</span>
              </div>
            )}
          </div>

          <div style={{
            border: "1px solid #cbd5e1",
            borderRadius: 8,
            background: "#fff",
            padding: 16,
            minHeight: 86,
          }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#64748b", marginBottom: 8 }}>
              Why?
            </div>
            <div style={{ color: result ? "#1f2937" : "#94a3b8", fontSize: 14, lineHeight: 1.55 }}>
              {result ? result.why : "The explanation will appear after scoring."}
            </div>
          </div>
        </div>

        {error && (
          <div style={{ color: "#b91c1c", background: "#fee2e2", borderRadius: 8, padding: 12, marginTop: 14, fontSize: 13 }}>
            {error.message}
          </div>
        )}
      </section>
    </div>
  );
}
