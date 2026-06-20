FROM rust:1.82-slim AS execution

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

RUN cargo build --release

CMD ["./target/release/octopus-execution"]