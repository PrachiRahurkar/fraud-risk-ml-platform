import express from "express";
import { ApolloServer } from "@apollo/server";
import { expressMiddleware } from "@apollo/server/express4";
import { typeDefs } from "./schema/typeDefs";
import { fundResolvers } from "./resolvers/fund";
import { feedbackResolvers } from "./resolvers/feedback";
import { metricsResolvers } from "./resolvers/metrics";

const PORT = parseInt(process.env.PORT || "4000");

const resolvers = {
  Query: {
    ...fundResolvers.Query,
    ...metricsResolvers.Query,
  },
  Mutation: {
    ...feedbackResolvers.Mutation,
  },
};

async function main() {
  const app = express();
  app.use(express.json());

  const server = new ApolloServer({ typeDefs, resolvers });
  await server.start();

  app.use("/graphql", expressMiddleware(server));

  app.listen(PORT, () => {
    console.log(`BFF GraphQL server running at http://localhost:${PORT}/graphql`);
  });
}

main().catch(console.error);
