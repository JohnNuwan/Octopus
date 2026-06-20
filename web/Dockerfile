FROM node:22-alpine AS web

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY src/ ./src/

EXPOSE 3000

CMD ["npm", "start"]