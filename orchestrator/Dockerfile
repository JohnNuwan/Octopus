FROM golang:1.23-alpine AS orchestrator

WORKDIR /app

# Install protoc + Go plugins
RUN apk add --no-cache protobuf protobuf-dev && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

COPY go.mod ./
RUN go mod download 2>&1 || echo "⚠️ go.sum sera généré au build"

# Générer les stubs gRPC depuis le proto
COPY proto/ ./proto/
RUN protoc --go_out=. --go_opt=paths=source_relative \
           --go-grpc_out=. --go-grpc_opt=paths=source_relative \
           proto/octopus.proto

COPY cmd/ ./cmd/
COPY internal/ ./internal/

RUN go build -o orchestrator ./cmd/

EXPOSE 9091 8080

CMD ["./orchestrator"]