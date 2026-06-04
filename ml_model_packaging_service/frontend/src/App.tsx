import React, { useState } from "react";
import { ApolloClient, InMemoryCache, ApolloProvider } from "@apollo/client";
import { FraudQueue } from "./pages/FraudQueue";
import { ModelMetrics } from "./pages/ModelMetrics";
import { Explainability } from "./pages/Explainability";

const client = new ApolloClient({
  uri: import.meta.env.VITE_GRAPHQL_URL || "http://localhost:4000/graphql",
  cache: new InMemoryCache(),
});

type Page = "queue" | "metrics" | "explain";

const NAV: { key: Page; label: string }[] = [
  { key: "queue", label: "Fraud Queue" },
  { key: "metrics", label: "Model Metrics" },
  { key: "explain", label: "Explainability" },
];

export default function App() {
  const [page, setPage] = useState<Page>("queue");

  return (
    <ApolloProvider client={client}>
      <div style={{ minHeight: "100vh", background: "#f8fafc", fontFamily: "system-ui, sans-serif" }}>
        <nav style={{
          background: "#1e293b", padding: "0 24px", display: "flex",
          alignItems: "center", gap: 24, height: 52
        }}>
          <span style={{ color: "#fff", fontWeight: 700, fontSize: 16, marginRight: 16 }}>
            Fraud Risk
          </span>
          {NAV.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setPage(key)}
              style={{
                background: "none", border: "none", cursor: "pointer",
                color: page === key ? "#818cf8" : "#94a3b8",
                fontWeight: page === key ? 700 : 400,
                fontSize: 14, padding: "4px 0",
                borderBottom: page === key ? "2px solid #818cf8" : "none",
              }}
            >
              {label}
            </button>
          ))}
        </nav>

        <main style={{ padding: "24px 0" }}>
          {page === "queue" && <FraudQueue />}
          {page === "metrics" && <ModelMetrics />}
          {page === "explain" && <Explainability />}
        </main>
      </div>
    </ApolloProvider>
  );
}
