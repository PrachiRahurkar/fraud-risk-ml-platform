import { batchPredict, getExplanation, predictSingle } from "../clients/mlApi";

// In production this would query a data store; here we load the test CSV inline.
// Replace with your actual data source (DB, GCS, etc.)
const SAMPLE_FUNDS: any[] = []; // populated at startup from CSV or cache

function stableFundId(text: string): number {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash % 1_000_000_000);
}

function campaignTitle(text: string): string {
  const firstLine = text.split(/\r?\n/).map(line => line.trim()).find(Boolean);
  return (firstLine || "User submitted fundraiser campaign").slice(0, 160);
}

function mapFeatures(features: any[] = []) {
  return features.map((f: any) => ({
    name: f.name,
    shapValue: f.shap_value,
    direction: f.direction,
  }));
}

function pct(value: number | null | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "unavailable";
}

function buildWhy(prediction: any, requestedMode: string): string {
  const labelText = prediction.label === 1 ? "Fraud" : "Legit";
  const relation = prediction.label === 1 ? "at or above" : "below";
  const parts = [
    `Labeled ${labelText} because the final score was ${pct(prediction.fraud_score)}, ${relation} the ${pct(prediction.threshold)} threshold.`,
  ];

  if (requestedMode === "ENSEMBLE") {
    if (prediction.lora_score == null) {
      parts.push("LoRA was selected, but the adapter was not loaded, so the API returned an XGBoost-only score.");
    } else {
      parts.push(`The ensemble combined XGBoost at ${pct(prediction.xgb_score)} with the LoRA text model at ${pct(prediction.lora_score)} using a ${(prediction.lora_weight * 100).toFixed(0)}% LoRA weight.`);
    }
  } else {
    parts.push(`Only the XGBoost tabular model was used; its score was ${pct(prediction.xgb_score)}.`);
  }

  const topFeatures = mapFeatures(prediction.top_features).slice(0, 3);
  if (topFeatures.length > 0) {
    const featureText = topFeatures
      .map((f: any) => `${f.name} (${f.direction}, ${f.shapValue.toFixed(3)})`)
      .join(", ");
    parts.push(`Top stored signals: ${featureText}.`);
  } else {
    parts.push("No stored SHAP row exists for this new campaign text, so the explanation uses model component scores rather than per-feature SHAP values.");
  }

  return parts.join(" ");
}

export const fundResolvers = {
  Query: {
    fraudQueue: async (_: any, { minScore = 0.7, limit = 50 }: any) => {
      if (SAMPLE_FUNDS.length === 0) return [];
      const predictions = await batchPredict(SAMPLE_FUNDS.slice(0, 200));
      return predictions
        .filter((p: any) => p.fraud_score >= minScore)
        .sort((a: any, b: any) => b.fraud_score - a.fraud_score)
        .slice(0, limit)
        .map((p: any) => ({
          fundId: String(p.fund_id),
          fraudScore: p.fraud_score,
          label: p.label,
          threshold: p.threshold,
          topFeatures: (p.top_features || []).map((f: any) => ({
            name: f.name,
            shapValue: f.shap_value,
            direction: f.direction,
          })),
        }));
    },

    fundExplanation: async (_: any, { fundId }: { fundId: string }) => {
      const data = await getExplanation(fundId);
      return {
        fundId,
        fraudScore: data.fraud_score ?? null,
        topFeatures: (data.top_features || []).map((f: any) => ({
          name: f.name,
          shapValue: f.shap_value,
          direction: f.direction,
        })),
      };
    },
  },

  Mutation: {
    scoreCampaign: async (
      _: any,
      { campaignText, modelMode }: { campaignText: string; modelMode: "XGB" | "ENSEMBLE" }
    ) => {
      const text = campaignText.trim();
      const title = campaignTitle(text);
      const prediction = await predictSingle({
        fund_id: stableFundId(text),
        title,
        description: text,
        title_len: title.length,
        descr_len: text.length,
        primary_phone_checks__line_type: "unknown",
        model_mode: modelMode === "ENSEMBLE" ? "ensemble" : "xgb",
      });
      const resolvedMode = prediction.model_mode === "ensemble" ? "ENSEMBLE" : "XGB";

      return {
        fraudScore: prediction.fraud_score,
        label: prediction.label,
        labelText: prediction.label === 1 ? "Fraud" : "Legit",
        threshold: prediction.threshold,
        modelMode: resolvedMode,
        xgbScore: prediction.xgb_score ?? null,
        loraScore: prediction.lora_score ?? null,
        loraWeight: prediction.lora_weight ?? 0,
        topFeatures: mapFeatures(prediction.top_features),
        why: buildWhy(prediction, modelMode),
      };
    },
  },
};
