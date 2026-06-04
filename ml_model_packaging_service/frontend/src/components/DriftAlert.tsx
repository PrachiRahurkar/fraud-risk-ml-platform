import React from "react";

interface Props {
  driftDetected: boolean;
  driftedFeatures: string[];
  generatedAt: string;
}

export function DriftAlert({ driftDetected, driftedFeatures, generatedAt }: Props) {
  if (!driftDetected) {
    return (
      <div style={{
        padding: 12, borderRadius: 8, background: "#dcfce7",
        color: "#15803d", fontWeight: 600, fontSize: 13
      }}>
        No data drift detected as of {new Date(generatedAt).toLocaleString()}
      </div>
    );
  }

  return (
    <div style={{
      padding: 12, borderRadius: 8, background: "#fee2e2",
      color: "#b91c1c", fontSize: 13
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>
        Data drift detected — {driftedFeatures.length} feature(s) shifted
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {driftedFeatures.map(f => (
          <span key={f} style={{
            background: "#fca5a5", borderRadius: 4, padding: "1px 8px",
            fontWeight: 600, fontSize: 11,
          }}>{f}</span>
        ))}
      </div>
      <div style={{ color: "#7f1d1d", fontSize: 11, marginTop: 6 }}>
        Last checked: {new Date(generatedAt).toLocaleString()}
      </div>
    </div>
  );
}
