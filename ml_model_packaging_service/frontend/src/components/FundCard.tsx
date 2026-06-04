import React from "react";
import { useMutation } from "@apollo/client";
import { ScoreGauge } from "./ScoreGauge";
import { SUBMIT_REVIEW } from "../graphql/queries";

interface Feature {
  name: string;
  shapValue: number;
  direction: string;
}

interface Props {
  fundId: string;
  fraudScore: number;
  threshold: number;
  topFeatures: Feature[];
  onReviewed?: () => void;
}

export function FundCard({ fundId, fraudScore, threshold, topFeatures, onReviewed }: Props) {
  const [submitReview, { loading }] = useMutation(SUBMIT_REVIEW);

  const handleReview = async (isFraud: boolean) => {
    await submitReview({ variables: { fundId, isFraud, confidence: 1.0 } });
    onReviewed?.();
  };

  return (
    <div style={{
      border: "1px solid #e5e7eb", borderRadius: 8, padding: 16,
      marginBottom: 12, background: "#fff", boxShadow: "0 1px 3px rgba(0,0,0,.05)"
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: 15 }}>Fund #{fundId}</div>
          <div style={{ fontSize: 12, color: "#6b7280", marginTop: 2 }}>
            {topFeatures.slice(0, 2).map(f => `${f.name}: ${f.shapValue.toFixed(3)}`).join(" · ")}
          </div>
        </div>
        <ScoreGauge score={fraudScore} threshold={threshold} />
      </div>

      <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
        {topFeatures.map(f => (
          <span key={f.name} style={{
            padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 600,
            background: f.direction === "fraud" ? "#fee2e2" : "#dcfce7",
            color: f.direction === "fraud" ? "#b91c1c" : "#15803d",
          }}>
            {f.name}
          </span>
        ))}
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
        <button
          onClick={() => handleReview(true)}
          disabled={loading}
          style={{
            flex: 1, padding: "6px 0", borderRadius: 6, border: "none",
            background: "#ef4444", color: "#fff", fontWeight: 600, cursor: "pointer",
          }}
        >
          Flag Fraud
        </button>
        <button
          onClick={() => handleReview(false)}
          disabled={loading}
          style={{
            flex: 1, padding: "6px 0", borderRadius: 6, border: "1px solid #d1d5db",
            background: "#f9fafb", color: "#374151", fontWeight: 600, cursor: "pointer",
          }}
        >
          Approve
        </button>
      </div>
    </div>
  );
}
