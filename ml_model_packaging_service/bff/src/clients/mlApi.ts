import axios from "axios";

const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";

const client = axios.create({ baseURL: ML_API_URL, timeout: 600_000 });

export async function predictSingle(record: object): Promise<any> {
  const { data } = await client.post("/predict", record);
  return data;
}

export async function batchPredict(records: object[]): Promise<any[]> {
  const { data } = await client.post("/predict/batch", { records });
  return data.predictions;
}

export async function getExplanation(fundId: string): Promise<any> {
  const { data } = await client.get(`/explain/${fundId}`);
  return data;
}

export async function getHealth(): Promise<any> {
  const { data } = await client.get("/health");
  return data;
}
