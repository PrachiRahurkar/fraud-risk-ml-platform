FROM node:20-slim

WORKDIR /app

COPY package.json ./
RUN npm install

COPY src/ ./src/
COPY ../proto/ ./proto/

RUN npm run build

EXPOSE 4000
CMD ["node", "dist/index.js"]
