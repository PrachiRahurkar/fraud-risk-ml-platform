FROM node:20-slim AS build

WORKDIR /app

ARG VITE_GRAPHQL_URL=http://localhost:4000/graphql
ENV VITE_GRAPHQL_URL=$VITE_GRAPHQL_URL

COPY ml_model_packaging_service/frontend/package.json ./
RUN npm install

COPY ml_model_packaging_service/frontend/index.html ./
COPY ml_model_packaging_service/frontend/tsconfig.json ./
COPY ml_model_packaging_service/frontend/tsconfig.node.json ./
COPY ml_model_packaging_service/frontend/vite.config.ts ./
COPY ml_model_packaging_service/frontend/src/ ./src/

RUN npm run build

FROM nginx:1.27-alpine

ENV PORT=8080

COPY ml_model_packaging_service/frontend/nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
