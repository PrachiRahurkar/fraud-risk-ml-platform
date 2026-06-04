import React from "react";
import { useQuery } from "@apollo/client";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell
} from "recharts";
import { GET_MODEL_METRICS, GET_DRIFT_REPORT } from "../graphql/queries";
import { DriftAlert } from "../components/DriftAlert";

const METRIC_LABELS: Record<string, string> = {
  aucRoc: "AUC-ROC",
  prAuc: "PR-AUC",
  f1: "F1",
  precisionAt100: "Prec@100",
  precisionAt500: "Prec@500",
};

export function ModelMetrics() {
  const { data: metricsData, loading: mLoading } = useQuery(GET_MODEL_METRICS, {
    variables: { window: "7d" },
  });
  const { data: driftData, loading: dLoading } = useQuery(GET_DRIFT_REPORT);

  const metrics = metricsData?.modelMetrics;
  const chartData = metrics
    ? Object.entries(METRIC_LABELS).map(([key, label]) => ({
        name: label,
        value: parseFloat((metrics[key] ?? 0).toFixed(4)),
      }))
    : [];

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 24 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Model Performance</h1>
      <p style={{ color: "#6b7280", marginBottom: 20, fontSize: 13 }}>
        Live metrics from the last 7 days of predictions.
      </p>

      {!dLoading && driftData?.driftReport && (
        <div style={{ marginBottom: 20 }}>
          <DriftAlert
            driftDetected={driftData.driftReport.driftDetected}
            driftedFeatures={driftData.driftReport.driftedFeatures}
            generatedAt={driftData.driftReport.generatedAt}
          />
        </div>
      )}

      {mLoading && <div style={{ color: "#6b7280" }}>Loading metrics…</div>}

      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={["#6366f1","#8b5cf6","#ec4899","#f97316","#14b8a6"][i % 5]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}

      {metrics && (
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 24
        }}>
          {Object.entries(METRIC_LABELS).map(([key, label]) => (
            <div key={key} style={{
              padding: 16, borderRadius: 8, background: "#f8fafc",
              border: "1px solid #e5e7eb", textAlign: "center"
            }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: "#1e293b" }}>
                {(metrics[key] ?? 0).toFixed(4)}
              </div>
              <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>{label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
