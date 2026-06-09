import { gql } from "@apollo/client";

export const GET_FRAUD_QUEUE = gql`
  query FraudQueue($minScore: Float, $limit: Int) {
    fraudQueue(minScore: $minScore, limit: $limit) {
      fundId
      fraudScore
      label
      threshold
      topFeatures {
        name
        shapValue
        direction
      }
    }
  }
`;

export const GET_MODEL_METRICS = gql`
  query ModelMetrics($window: String) {
    modelMetrics(window: $window) {
      aucRoc
      prAuc
      f1
      precisionAt100
      precisionAt500
      rollingAuc
      window
    }
  }
`;

export const GET_DRIFT_REPORT = gql`
  query DriftReport {
    driftReport {
      driftDetected
      driftedFeatures
      shareOfDriftedFeatures
      generatedAt
    }
  }
`;

export const GET_EXPLANATION = gql`
  query FundExplanation($fundId: ID!) {
    fundExplanation(fundId: $fundId) {
      fundId
      fraudScore
      topFeatures {
        name
        shapValue
        direction
      }
    }
  }
`;

export const SCORE_CAMPAIGN = gql`
  mutation ScoreCampaign($campaignText: String!, $modelMode: ModelMode!) {
    scoreCampaign(campaignText: $campaignText, modelMode: $modelMode) {
      fraudScore
      label
      labelText
      threshold
      modelMode
      xgbScore
      loraScore
      loraWeight
      why
      topFeatures {
        name
        shapValue
        direction
      }
    }
  }
`;

export const SUBMIT_REVIEW = gql`
  mutation SubmitReview($fundId: ID!, $isFraud: Boolean!, $confidence: Float, $notes: String) {
    submitReview(fundId: $fundId, isFraud: $isFraud, confidence: $confidence, notes: $notes)
  }
`;
