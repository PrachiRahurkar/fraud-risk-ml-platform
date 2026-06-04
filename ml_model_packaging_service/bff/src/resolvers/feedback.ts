import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import path from "path";

const PROTO_PATH = path.join(__dirname, "../../../proto/feedback.proto");
const FEEDBACK_SVC_URL = process.env.FEEDBACK_SVC_URL || "localhost:50051";

const packageDef = protoLoader.loadSync(PROTO_PATH, { keepCase: true });
const proto: any = grpc.loadPackageDefinition(packageDef).feedback;

const client = new proto.FeedbackService(
  FEEDBACK_SVC_URL,
  grpc.credentials.createInsecure()
);

function submitLabelGrpc(req: {
  fund_id: number;
  is_fraud: boolean;
  confidence: number;
  reviewer_id: string;
  notes: string;
}): Promise<{ success: boolean }> {
  return new Promise((resolve, reject) => {
    client.SubmitLabel(req, (err: any, response: any) => {
      if (err) reject(err);
      else resolve(response);
    });
  });
}

export const feedbackResolvers = {
  Mutation: {
    submitReview: async (
      _: any,
      { fundId, isFraud, confidence = 1.0, notes = "" }: any
    ) => {
      try {
        const result = await submitLabelGrpc({
          fund_id: parseInt(fundId),
          is_fraud: isFraud,
          confidence,
          reviewer_id: "analyst",
          notes,
        });
        return result.success;
      } catch (e) {
        console.error("Feedback gRPC error:", e);
        return false;
      }
    },
  },
};
