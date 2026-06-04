import React from "react";

interface Props {
  score: number;
  threshold: number;
}

export function ScoreGauge({ score, threshold }: Props) {
  const pct = Math.round(score * 100);
  const color = score >= threshold ? "#ef4444" : score >= threshold * 0.7 ? "#f97316" : "#22c55e";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: "50%",
          background: `conic-gradient(${color} ${pct * 3.6}deg, #e5e7eb 0deg)`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 11,
          fontWeight: 700,
          color: "#111",
        }}
      >
        <div style={{ background: "#fff", borderRadius: "50%", width: 32, height: 32,
          display: "flex", alignItems: "center", justifyContent: "center" }}>
          {pct}%
        </div>
      </div>
    </div>
  );
}
