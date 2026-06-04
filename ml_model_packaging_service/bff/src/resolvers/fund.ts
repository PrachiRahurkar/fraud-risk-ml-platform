import { batchPredict, getExplanation } from "../clients/mlApi";

// In production this would query a data store; here we load the test CSV inline.
// Replace with your actual data source (DB, GCS, etc.)
const SAMPLE_FUNDS: any[] = []; // populated at startup from CSV or cache

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
};
