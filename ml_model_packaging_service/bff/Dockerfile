FROM node:20-slim

WORKDIR /app/ml_model_packaging_service/bff

COPY ml_model_packaging_service/bff/package.json ./
RUN npm install

COPY ml_model_packaging_service/bff/tsconfig.json ./
COPY ml_model_packaging_service/bff/src/ ./src/
COPY ml_model_packaging_service/proto/ ../proto/
COPY ml_observability_monitoring/proto/ ../../ml_observability_monitoring/proto/

RUN npm run build

EXPOSE 4000
CMD ["node", "dist/index.js"]
