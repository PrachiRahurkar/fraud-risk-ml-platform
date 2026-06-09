import express from "express";
import { ApolloServer } from "@apollo/server";
import { expressMiddleware } from "@apollo/server/express4";
import { typeDefs } from "./schema/typeDefs";
import { fundResolvers } from "./resolvers/fund";
import { feedbackResolvers } from "./resolvers/feedback";
import { metricsResolvers } from "./resolvers/metrics";

const PORT = parseInt(process.env.PORT || "4000");
const CORS_ORIGIN = process.env.CORS_ORIGIN || "*";

const resolvers = {
  Query: {
    ...fundResolvers.Query,
    ...metricsResolvers.Query,
  },
  Mutation: {
    ...fundResolvers.Mutation,
    ...feedbackResolvers.Mutation,
  },
};

async function main() {
  const app = express();
  app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", CORS_ORIGIN);
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    if (req.method === "OPTIONS") {
      res.sendStatus(204);
      return;
    }
    next();
  });
  app.use(express.json());

  const server = new ApolloServer({ typeDefs, resolvers });
  await server.start();

  app.use("/graphql", expressMiddleware(server));

  app.listen(PORT, () => {
    console.log(`BFF GraphQL server running at http://localhost:${PORT}/graphql`);
  });
}

main().catch(console.error);
