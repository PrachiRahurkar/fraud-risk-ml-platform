import React, { useState } from "react";
import { useQuery } from "@apollo/client";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from "recharts";
import { GET_EXPLANATION } from "../graphql/queries";

export function Explainability() {
  const [fundId, setFundId] = useState("");
  const [queryId, setQueryId] = useState<string | null>(null);

  const { data, loading, error } = useQuery(GET_EXPLANATION, {
    variables: { fundId: queryId },
    skip: !queryId,
  });

  const features = data?.fundExplanation?.topFeatures ?? [];
  const chartData = features.map((f: any) => ({
    name: f.name,
    value: Math.abs(f.shapValue),
    raw: f.shapValue,
    direction: f.direction,
  }));

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: 24 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Prediction Explainability</h1>
      <p style={{ color: "#6b7280", marginBottom: 20, fontSize: 13 }}>
        SHAP feature contributions for a specific fundraiser.
      </p>

      <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        <input
          type="text"
          placeholder="Enter fund_id (e.g. 62614939)"
          value={fundId}
          onChange={e => setFundId(e.target.value)}
          style={{
            flex: 1, padding: "8px 12px", borderRadius: 6,
            border: "1px solid #d1d5db", fontSize: 14,
          }}
        />
        <button
          onClick={() => setQueryId(fundId)}
          style={{
            padding: "8px 18px", borderRadius: 6, background: "#6366f1",
            color: "#fff", border: "none", fontWeight: 600, cursor: "pointer",
          }}
        >
          Explain
        </button>
      </div>

      {loading && <div style={{ color: "#6b7280" }}>Loading explanation…</div>}
      {error && <div style={{ color: "#ef4444" }}>No explanation found for this fund_id.</div>}

      {chartData.length > 0 && (
        <>
          <div style={{ marginBottom: 12, fontSize: 13, color: "#374151" }}>
            <strong>Fraud score:</strong> {(data?.fundExplanation?.fraudScore ?? 0).toFixed(4)}
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              layout="vertical"
              data={chartData}
              margin={{ top: 8, right: 24, left: 120, bottom: 0 }}
            >
              <XAxis type="number" tick={{ fontSize: 11 }} domain={[0, "auto"]} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} width={120} />
              <Tooltip
                formatter={(v: number, _: string, entry: any) =>
                  [`${entry.payload.raw.toFixed(4)} (${entry.payload.direction})`, "SHAP"]}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {chartData.map((entry: any, i: number) => (
                  <Cell
                    key={i}
                    fill={entry.direction === "fraud" ? "#ef4444" : "#22c55e"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 8 }}>
            Red = increases fraud probability · Green = decreases fraud probability
          </div>
        </>
      )}
    </div>
  );
}
