import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import path from "path";

const PROTO_PATH = path.join(__dirname, "../../../../ml_observability_monitoring/proto/monitoring.proto");
const MONITORING_SVC_URL = process.env.MONITORING_SVC_URL || "localhost:50052";

const packageDef = protoLoader.loadSync(PROTO_PATH, { keepCase: true });
const proto: any = grpc.loadPackageDefinition(packageDef).monitoring;

const monitorClient = new proto.MonitoringService(
  MONITORING_SVC_URL,
  grpc.credentials.createInsecure()
);

function getRollingMetrics(window: string): Promise<any> {
  return new Promise((resolve, reject) => {
    monitorClient.GetRollingMetrics({ window }, (err: any, res: any) => {
      if (err) reject(err);
      else resolve(res);
    });
  });
}

function getDriftReport(): Promise<any> {
  return new Promise((resolve, reject) => {
    monitorClient.GetDriftReport({}, (err: any, res: any) => {
      if (err) reject(err);
      else resolve(res);
    });
  });
}

export const metricsResolvers = {
  Query: {
    modelMetrics: async (_: any, { window = "7d" }: any) => {
      try {
        const m = await getRollingMetrics(window);
        return {
          aucRoc: m.auc_roc,
          prAuc: m.pr_auc,
          f1: m.f1,
          precisionAt100: m.precision_at_100,
          precisionAt500: m.precision_at_500,
          rollingAuc: m.rolling_auc,
          window,
        };
      } catch {
        return null;
      }
    },

    driftReport: async () => {
      try {
        const d = await getDriftReport();
        return {
          driftDetected: d.drift_detected,
          driftedFeatures: d.drifted_features || [],
          shareOfDriftedFeatures: d.share_of_drifted_features,
          generatedAt: d.generated_at,
        };
      } catch {
        return {
          driftDetected: false,
          driftedFeatures: [],
          generatedAt: new Date().toISOString(),
        };
      }
    },
  },
};
