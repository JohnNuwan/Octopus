// Octopus Execution Engine — Moteur d'exécution MT5
// Point d'entrée principal. Se connecte à l'orchestrateur, attend les décisions,
// exécute les ordres MT5, et envoie les confirmations.
//
// Main entry point. Connects to the orchestrator, awaits trading decisions,
// executes MT5 orders, and sends confirmations.

mod client;
mod executor;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use client::Client;
use executor::{MT5Config, MT5Connector};

// ─── CLI Arguments ──────────────────────────────────────────────────────────

/// Octopus Execution Engine — MT5 trade executor with gRPC orchestration
#[derive(Parser, Debug)]
#[command(name = "octopus-execution")]
#[command(about = "MT5 order execution engine for the Octopus trading system")]
struct Args {
    /// Adresse de l'orchestrateur (host:port)
    /// Orchestrator gRPC endpoint
    #[arg(long, default_value = "localhost:9091")]
    orchestrator_addr: String,

    /// Login MT5
    #[arg(long, default_value = "")]
    mt5_login: String,

    /// Mot de passe MT5
    /// MT5 password
    #[arg(long, default_value = "")]
    mt5_password: String,

    /// Serveur MT5
    /// MT5 server name
    #[arg(long, default_value = "ICMarkets-Demo")]
    mt5_server: String,

    /// Symbole de trading (défaut: XAUUSD)
    /// Trading symbol
    #[arg(long, default_value = "XAUUSD")]
    symbol: String,

    /// Intervalle de heartbeat en secondes
    /// Heartbeat interval in seconds
    #[arg(long, default_value_t = 30)]
    heartbeat_interval: u64,
}

// ─── Signal Handling ────────────────────────────────────────────────────────

fn setup_signal_handler() -> Arc<AtomicBool> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to listen for SIGINT");
        info!("SIGINT received — initiating graceful shutdown");
        r.store(false, Ordering::Relaxed);
    });

    running
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build the orchestrator gRPC address from env vars or defaults.
/// ORCHESTRATOR_HOST / ORCHESTRATOR_PORT override CLI / defaults.
fn build_orchestrator_addr(cli_addr: &str) -> String {
    let host = std::env::var("ORCHESTRATOR_HOST")
        .unwrap_or_else(|_| {
            // Extract host from cli_addr if it contains a colon, otherwise use cli_addr
            if let Some((h, _)) = cli_addr.split_once(':') {
                h.to_string()
            } else {
                cli_addr.to_string()
            }
        });
    let port = std::env::var("ORCHESTRATOR_PORT")
        .unwrap_or_else(|_| {
            // Extract port from cli_addr if it contains a colon, otherwise use "9091"
            if let Some((_, p)) = cli_addr.split_once(':') {
                p.to_string()
            } else {
                "9091".to_string()
            }
        });
    format!("http://{}:{}", host, port)
}

// ─── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Initialisation du logging structuré
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,octopus_execution=debug")),
        )
        .init();

    let args = Args::parse();
    let running = setup_signal_handler();

    info!(
        "Octopus Execution Engine starting — symbol={}",
        args.symbol
    );

    // ── Configuration MT5 ───────────────────────────────────────────────────────
    let mt5_config = MT5Config {
        login: args.mt5_login,
        password: args.mt5_password,
        server: args.mt5_server,
        symbol: args.symbol,
    };

    // ── Connexion MT5 ───────────────────────────────────────────────────────────
    let mut connector = MT5Connector::new(mt5_config);
    connector
        .connect()
        .await
        .context("Failed to connect to MT5")?;

    info!("MT5 connector initialized successfully");

    // ── Connexion à l'orchestrateur ─────────────────────────────────────────────
    let addr = build_orchestrator_addr(&args.orchestrator_addr);

    let mut client = Client::connect(addr.clone())
        .await
        .context("Failed to connect to orchestrator")?;

    let start_time = Utc::now();

    info!("Octopus Execution Engine started successfully");
    info!("Entering main loop — heartbeat every {}s", args.heartbeat_interval);

    // ── Boucle principale ──────────────────────────────────────────────────────
    while running.load(Ordering::Relaxed) {
        // Heartbeat vers l'orchestrateur
        match client.health_check().await {
            Ok(resp) => {
                if !resp.healthy {
                    warn!("Orchestrator reports unhealthy status");
                }
            }
            Err(e) => {
                error!("Health check failed: {}", e);
                // Tentative de reconnexion si le heartbeat échoue
                warn!("Attempting to reconnect to orchestrator...");

                match Client::connect(addr.clone()).await {
                    Ok(new_client) => {
                        client = new_client;
                        info!("Reconnected to orchestrator successfully");
                    }
                    Err(re_err) => {
                        error!("Failed to reconnect: {}", re_err);
                    }
                }
            }
        }

        // Attente avant le prochain cycle
        // Dans une version future, ceci sera remplacé par un streaming gRPC
        // pour recevoir les décisions en temps réel.
        tokio::time::sleep(tokio::time::Duration::from_secs(
            args.heartbeat_interval,
        ))
        .await;
    }

    // ── Arrêt gracieux ─────────────────────────────────────────────────────────
    info!("Shutting down Octopus Execution Engine...");

    if let Err(e) = connector.disconnect().await {
        warn!("Error disconnecting MT5: {}", e);
    }

    let uptime = Utc::now() - start_time;
    info!(
        "Octopus Execution Engine stopped — uptime={}s",
        uptime.num_seconds()
    );

    Ok(())
}