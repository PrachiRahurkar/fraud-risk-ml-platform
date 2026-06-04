import React, { useState } from "react";
import { useQuery } from "@apollo/client";
import { GET_FRAUD_QUEUE } from "../graphql/queries";
import { FundCard } from "../components/FundCard";

export function FraudQueue() {
  const [minScore, setMinScore] = useState(0.7);
  const { data, loading, error, refetch } = useQuery(GET_FRAUD_QUEUE, {
    variables: { minScore, limit: 50 },
    fetchPolicy: "network-only",
  });

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: 24 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Fraud Review Queue</h1>
      <p style={{ color: "#6b7280", marginBottom: 20, fontSize: 13 }}>
        Fundraisers flagged above the risk threshold. Review and label each one.
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 20 }}>
        <label style={{ fontSize: 13, color: "#374151" }}>
          Min fraud score:
          <input
            type="range" min={0.5} max={0.99} step={0.01} value={minScore}
            onChange={e => setMinScore(parseFloat(e.target.value))}
            style={{ marginLeft: 8 }}
          />
          <strong style={{ marginLeft: 6 }}>{(minScore * 100).toFixed(0)}%</strong>
        </label>
        <button
          onClick={() => refetch()}
          style={{
            padding: "4px 14px", borderRadius: 6, border: "1px solid #d1d5db",
            background: "#f3f4f6", cursor: "pointer", fontSize: 12,
          }}
        >
          Refresh
        </button>
      </div>

      {loading && <div style={{ color: "#6b7280" }}>Loading queue…</div>}
      {error && <div style={{ color: "#ef4444" }}>Error: {error.message}</div>}

      {data?.fraudQueue?.length === 0 && (
        <div style={{ color: "#6b7280", fontSize: 13 }}>
          No funds above {(minScore * 100).toFixed(0)}% threshold.
        </div>
      )}

      {data?.fraudQueue?.map((fund: any) => (
        <FundCard
          key={fund.fundId}
          fundId={fund.fundId}
          fraudScore={fund.fraudScore}
          threshold={fund.threshold}
          topFeatures={fund.topFeatures}
          onReviewed={() => refetch()}
        />
      ))}
    </div>
  );
}
