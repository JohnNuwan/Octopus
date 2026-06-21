// Client gRPC pour la communication avec l'orchestrateur Octopus
// Octopus gRPC client — connects to the orchestrator for trading decisions

use anyhow::{Context, Result};
use tonic::transport::Channel;
use tracing::{info, error, warn};

// Generated proto types — compiled by tonic-build from proto/octopus.proto
pub mod octopus {
    tonic::include_proto!("octopus");
}

use octopus::orchestrator_client::OrchestratorClient;
use octopus::{
    ConfirmationAck, HealthRequest, HealthResponse,
    TradingDecision, DecisionAck, OrderConfirmation,
};

/// Client gRPC pour l'orchestrateur Octopus
/// Handles all bidirectional communication with the central orchestrator
#[allow(dead_code)]
pub struct Client {
    inner: OrchestratorClient<Channel>,
}

impl Client {
    /// Connecte au service Orchestrator à l'adresse donnée
    /// Connect to the Orchestrator gRPC service at the given address
    pub async fn connect(addr: String) -> Result<Self> {
        let channel = Channel::from_shared(addr.clone())
            .context(format!("Invalid orchestrator address: {}", addr))?
            .connect()
            .await
            .context(format!("Failed to connect to orchestrator at {}", addr))?;

        info!("Connected to orchestrator at {}", addr);
        Ok(Self {
            inner: OrchestratorClient::new(channel),
        })
    }

    /// Envoie une décision de trading à l'orchestrateur
    /// Send a trading decision to the orchestrator for validation/execution routing
    pub async fn send_trading_decision(
        &mut self,
        decision: TradingDecision,
    ) -> Result<DecisionAck> {
        let ack = self
            .inner
            .send_trading_decision(decision)
            .await
            .context("Failed to send trading decision")?
            .into_inner();

        if ack.accepted {
            info!("Trading decision accepted by orchestrator");
        } else {
            warn!(
                "Trading decision rejected by orchestrator: {}",
                ack.reason
            );
        }

        Ok(ack)
    }

    /// Confirme l'exécution d'un ordre auprès de l'orchestrateur
    /// Confirm that an order has been executed back to the orchestrator
    pub async fn confirm_order_execution(
        &mut self,
        confirmation: OrderConfirmation,
    ) -> Result<ConfirmationAck> {
        let ack = self
            .inner
            .confirm_order_execution(confirmation)
            .await
            .context("Failed to confirm order execution")?
            .into_inner();

        info!(
            "Order execution confirmation sent, received={}",
            ack.received
        );
        Ok(ack)
    }

    /// Vérifie l'état de santé de l'orchestrateur
    /// Check if the orchestrator service is healthy
    pub async fn health_check(&mut self) -> Result<HealthResponse> {
        let resp = self
            .inner
            .health_check(HealthRequest {})
            .await
            .context("Failed to perform health check")?
            .into_inner();

        if resp.healthy {
            info!(
                "Orchestrator health check OK (version={}, uptime={}s)",
                resp.version, resp.uptime_seconds
            );
        } else {
            error!("Orchestrator health check FAILED");
        }

        Ok(resp)
    }
}