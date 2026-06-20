// Octopus Orchestrator — Cerveau central du système de trading.
//
// Service Go responsable du routage gRPC entre les modules :
// - Engine (Python/JAX) → décisions de trading
// - Execution (Rust) → ordres MT5
// - Quant (Julia) → optimisations, risk management
// - Web (TypeScript) → dashboard monitoring

package main

import (
	"context"
	"log"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

const (
	defaultGRPCPort = ":9091"
	defaultHTTPPort = ":8080"
	version         = "0.2.0"
)

// Server représente l'orchestrateur central.
// Gère le routage des messages entre les services et le monitoring.
type Server struct {
	UnimplementedOrchestratorServer

	mu       sync.RWMutex
	clients  map[string]*ClientState // clients connectés (engine, execution, quant, web)
	uptime   time.Time
	stopCh   chan struct{}
}

// ClientState décrit l'état d'un client connecté.
type ClientState struct {
	ID          string
	ServiceType string // engine, execution, quant, web
	ConnectedAt time.Time
	LastSeen    time.Time
}

// NewServer crée une nouvelle instance du serveur orchestrator.
func NewServer() *Server {
	return &Server{
		clients: make(map[string]*ClientState),
		uptime:  time.Now(),
		stopCh:  make(chan struct{}),
	}
}

// HealthCheck vérifie l'état de santé du service.
func (s *Server) HealthCheck(ctx context.Context, req *HealthRequest) (*HealthResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return &HealthResponse{
		Healthy:       true,
		Version:       version,
		UptimeSeconds: int64(time.Since(s.uptime).Seconds()),
	}, nil
}

// SendTradingDecision reçoit une décision de l'Engine et la route vers l'Execution.
func (s *Server) SendTradingDecision(ctx context.Context, decision *TradingDecision) (*DecisionAck, error) {
	log.Printf("📊 Trading Decision: %s action=%d lots=%.2f confidence=%.2f",
		decision.Symbol, decision.Action, decision.Lots, decision.Confidence)

	// Vérification des règles FTMO (simplifiée)
	if decision.Lots <= 0 {
		return &DecisionAck{
			Accepted: false,
			Reason:   "Invalid lot size",
		}, nil
	}

	// Router vers le service d'exécution
	// (en production, appel gRPC vers le service Execution)
	execClient := s.getClientByType("execution")
	if execClient == nil {
		return &DecisionAck{
			Accepted: false,
			Reason:   "No execution client connected",
		}, nil
	}

	log.Printf("✅ Decision accepted — routed to execution client %s", execClient.ID)
	return &DecisionAck{Accepted: true}, nil
}

// ConfirmOrderExecution reçoit la confirmation d'un ordre exécuté.
func (s *Server) ConfirmOrderExecution(ctx context.Context, conf *OrderConfirmation) (*ConfirmationAck, error) {
	log.Printf("📈 Order confirmed: %s %s pnl=%.2f",
		conf.Symbol, conf.OrderId, conf.Pnl)
	return &ConfirmationAck{Received: true}, nil
}

// GetMarketData fournit des données de marché à l'Engine.
func (s *Server) GetMarketData(ctx context.Context, req *MarketDataRequest) (*MarketDataResponse, error) {
	// En production : interroge OANDA API / Dukascopy / Yahoo Finance
	log.Printf("📥 Market data request: %s %s (%d candles)",
		req.Symbol, req.Timeframe, req.NumCandles)

	return &MarketDataResponse{
		Symbol:  req.Symbol,
		Candles: nil, // Rempli par le service de données
	}, nil
}

// registerClient enregistre un nouveau client connecté.
func (s *Server) registerClient(id, serviceType string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.clients[id] = &ClientState{
		ID:          id,
		ServiceType: serviceType,
		ConnectedAt: time.Now(),
		LastSeen:    time.Now(),
	}

	log.Printf("🔌 Client connected: %s [%s]", id, serviceType)
}

// getClientByType retourne le premier client d'un type donné.
func (s *Server) getClientByType(serviceType string) *ClientState {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, c := range s.clients {
		if c.ServiceType == serviceType {
			return c
		}
	}
	return nil
}

// startGRPCServer démarre le serveur gRPC.
func startGRPCServer(addr string, server *Server) (*grpc.Server, net.Listener, error) {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, nil, err
	}

	grpcServer := grpc.NewServer()
	RegisterOrchestratorServer(grpcServer, server)

	// Reflection pour le debugging
	reflection.Register(grpcServer)

	go func() {
		log.Printf("🚀 gRPC server listening on %s", addr)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("gRPC server failed: %v", err)
		}
	}()

	return grpcServer, lis, nil
}

func main() {
	log.Printf("🐙 Octopus Orchestrator v%s — Starting...", version)
	log.Printf("   gRPC: %s | HTTP: %s", defaultGRPCPort, defaultHTTPPort)

	server := NewServer()

	// Démarrage du serveur gRPC
	grpcServer, lis, err := startGRPCServer(defaultGRPCPort, server)
	if err != nil {
		log.Fatalf("Failed to start gRPC server: %v", err)
	}
	defer lis.Close()

	// Graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Tick de monitoring
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	log.Println("✅ Orchestrator ready — waiting for clients...")

	for {
		select {
		case <-sigCh:
			log.Println("\n🛑 Shutting down...")
			grpcServer.GracefulStop()
			return

		case <-ticker.C:
			server.mu.RLock()
			clientCount := len(server.clients)
			server.mu.RUnlock()
			log.Printf("💚 Health check: %d clients connected, uptime %s",
				clientCount,
				time.Since(server.uptime).Round(time.Second))
		}
	}
}