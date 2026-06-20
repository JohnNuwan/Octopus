FROM golang:1.23-alpine AS orchestrator

WORKDIR /app

RUN apk add --no-cache protobuf-dev

COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ ./cmd/
COPY internal/ ./internal/
COPY proto/ ./proto/

RUN go build -o orchestrator ./cmd/

EXPOSE 9091 8080

CMD ["./orchestrator"]