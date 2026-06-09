export const typeDefs = `#graphql
  enum ModelMode {
    XGB
    ENSEMBLE
  }

  type FeatureContribution {
    name: String!
    shapValue: Float!
    direction: String!
  }

  type FundPrediction {
    fundId: ID!
    fraudScore: Float!
    label: Int!
    threshold: Float!
    topFeatures: [FeatureContribution!]!
  }

  type CampaignScore {
    fraudScore: Float!
    label: Int!
    labelText: String!
    threshold: Float!
    modelMode: ModelMode!
    xgbScore: Float
    loraScore: Float
    loraWeight: Float!
    topFeatures: [FeatureContribution!]!
    why: String!
  }

  type ModelMetrics {
    aucRoc: Float
    prAuc: Float
    f1: Float
    precisionAt100: Float
    precisionAt500: Float
    rollingAuc: Float
    window: String
  }

  type DriftReport {
    driftDetected: Boolean!
    driftedFeatures: [String!]!
    shareOfDriftedFeatures: Float
    generatedAt: String!
  }

  type Explanation {
    fundId: ID!
    fraudScore: Float
    topFeatures: [FeatureContribution!]!
  }

  type Query {
    fraudQueue(minScore: Float, limit: Int): [FundPrediction!]!
    fundExplanation(fundId: ID!): Explanation!
    modelMetrics(window: String): ModelMetrics!
    driftReport: DriftReport!
  }

  type Mutation {
    scoreCampaign(campaignText: String!, modelMode: ModelMode!): CampaignScore!
    submitReview(fundId: ID!, isFraud: Boolean!, confidence: Float, notes: String): Boolean!
  }
`;
