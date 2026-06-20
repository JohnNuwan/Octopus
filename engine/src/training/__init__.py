"""Boucle d'entraînement MuZero pour le moteur Octopus.

Orchestre l'entraînement complet du système :
1. Self-play pour collecter des données
2. Entraînement du World Model sur le replay buffer
3. Entraînement de l'Actor-Critic sur des rêves
4. Hybrid learning : intégration des trades live

References:
    Schrittwieser et al. "MuZero." Nature, 2020.
    Hafner et al. "DreamerV3." Nature, 2025.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Ajouter le chemin parent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .replay_buffer import ReplayBuffer, GameHistory
from ..networks.jepa import TSJEPA
from ..networks.world_model import RSSMWorldModel, symlog, symexp
from ..networks.actor_critic import ActorCritic
from ..mcts import MCTS
from ..environment import OctopusTradingEnv
from ..config import MuZeroConfig


class Trainer:
    """Entraîneur MuZero pour Octopus.
    
    Gère le cycle complet d'entraînement :
    collecte de données → mise à jour du modèle → évaluation.
    
    Attributes:
        config: Configuration globale.
        device: Device PyTorch (cpu/cuda).
        jepa: Module JEPA pour l'encodage.
        world_model: RSSM World Model.
        actor_critic: Module Actor-Critic.
        mcts: MCTS pour la planification.
        replay_buffer: Tampon de rejeu.
        optimizer: Optimiseur pour le réseau complet.
        step: Compteur global de pas d'entraînement.
    """
    
    def __init__(
        self,
        config: Optional[MuZeroConfig] = None,
        device: Optional[str] = None
    ) -> None:
        """Initialise l'entraîneur.
        
        Args:
            config: Configuration du moteur (utilise les valeurs par défaut si None).
            device: Device à utiliser ('cuda', 'cpu', ou None pour auto-détecter).
        """
        self.config = config or MuZeroConfig()
        
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Device: {self.device}")
        
        # Modules neurones
        self.jepa = TSJEPA(
            input_dim=self.config.observation_shape[0],
            seq_len=self.config.observation_shape[1],
            embedding_dim=self.config.jepa.embedding_dim
        ).to(self.device)
        
        self.world_model = RSSMWorldModel(
            stoch_size=self.config.world_model.stoch_size,
            stoch_classes=self.config.world_model.stoch_classes,
            deter_size=self.config.world_model.deter_size,
            hidden_dim=self.config.world_model.hidden_dim,
            action_dim=self.config.action_space_size,
            embedding_dim=self.config.jepa.embedding_dim
        ).to(self.device)
        
        self.actor_critic = ActorCritic(
            stoch_size=self.config.world_model.stoch_size,
            stoch_classes=self.config.world_model.stoch_classes,
            deter_size=self.config.world_model.deter_size,
            action_dim=self.config.action_space_size,
            gamma=self.config.actor_critic.gamma,
            lambda_=self.config.actor_critic.lambda_,
            entropy_coeff=self.config.actor_critic.entropy_coeff
        ).to(self.device)
        
        self.mcts = MCTS(
            action_dim=self.config.action_space_size,
            num_simulations=self.config.num_simulations,
            discount=self.config.discount,
            dirichlet_alpha=self.config.root_dirichlet_alpha,
            exploration_fraction=self.config.root_exploration_fraction
        )
        
        # Optimiseur
        self.optimizer = optim.AdamW(
            list(self.jepa.parameters()) +
            list(self.world_model.parameters()) +
            list(self.actor_critic.parameters()),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=500,
            gamma=0.7
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer_size
        )
        
        # Compteur
        self.step = 0
        self.best_reward = float('-inf')
        
        # Chemins
        self.checkpoint_dir = Path('weights')
        self.results_dir = Path('results')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, total_steps: Optional[int] = None) -> Dict[str, list]:
        """Lance l'entraînement complet.
        
        Args:
            total_steps: Nombre total de pas d'entraînement
                (utilise la configuration si None).
                
        Returns:
            Dictionnaire des métriques d'entraînement.
        """
        total_steps = total_steps or self.config.training_steps
        
        print(f"🚀 Début de l'entraînement : {total_steps} steps")
        print(f"   Symboles : {self.config.symbols}")
        print(f"   Batch size : {self.config.batch_size}")
        print(f"   Replay buffer : {self.config.replay_buffer_size}")
        print(f"   Checkpoint interval : {self.config.checkpoint_interval}")
        
        metrics_history = {
            'step': [],
            'reward': [],
            'policy_loss': [],
            'value_loss': [],
            'lr': [],
        }
        
        for step in range(total_steps):
            self.step = step
            
            # Self-play : collecter des données
            if step % 5 == 0 and hasattr(self, 'environments'):
                symbol = random.choice(self.config.symbols)
                reward = self._play_game(symbol)
                metrics_history['reward'].append(reward)
                metrics_history['step'].append(step)
            
            # Entraîner si le buffer est assez rempli
            if len(self.replay_buffer) >= self.config.batch_size:
                metrics = self._train_step()
                
                for k, v in metrics.items():
                    if k in metrics_history:
                        metrics_history[k].append(v)
                
                self.scheduler.step()
            
            # Log
            if step % self.config.checkpoint_interval == 0:
                self._log_progress(step, total_steps)
                self._save_checkpoint()
            
            # Hybrid learning : intégrer les trades live
            if step % 10 == 0:
                self._ingest_live_experience()
        
        # Sauvegarde finale
        self._save_checkpoint(is_best=False, suffix='_final')
        
        print("✅ Entraînement terminé")
        return metrics_history
    
    def _play_game(self, symbol: str) -> float:
        """Joue un épisode complet avec MCTS pour collecter des données.
        
        Args:
            symbol: Symbole à trader.
            
        Returns:
            Récompense totale de l'épisode.
        """
        self.jepa.eval()
        self.world_model.eval()
        
        env = self.environments.get(symbol)
        if env is None:
            return 0.0
        
        game = GameHistory()
        obs = env.reset()
        done = False
        
        while not done:
            # Encoder l'observation avec JEPA
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            embedding = self.jepa(obs_t)
            
            # Initial inference avec le World Model
            with torch.no_grad():
                stoch, deter = self.world_model.transition.forward(
                    prev_stoch=stoch if 'stoch' in dir() else torch.zeros(
                        1, self.config.world_model.stoch_size,
                        self.config.world_model.stoch_classes
                    ).to(self.device),
                    prev_action=prev_action if 'prev_action' in dir() else torch.zeros(
                        1, self.config.action_space_size
                    ).to(self.device),
                    prev_embedding=embedding
                )[0]
            
            # MCTS
            state_for_mcts = torch.cat([
                stoch.reshape(1, -1), deter
            ], dim=-1)
            
            visit_dist = self.mcts.search(
                state_for_mcts, self.world_model, self.device
            )
            
            action = np.random.choice(len(visit_dist), p=visit_dist)
            
            # Exécuter l'action
            next_obs, reward, done, _ = env.step(action)
            
            # Stocker l'expérience
            game.add_step(obs, action, reward, visit_dist, 0.0, done)
            
            obs = next_obs
        
        self.replay_buffer.add_game(game)
        
        return game.total_reward
    
    def _train_step(self) -> Dict[str, float]:
        """Effectue un pas d'entraînement sur le World Model et l'Actor-Critic.
        
        Returns:
            Dictionnaire des pertes.
        """
        self.jepa.train()
        self.world_model.train()
        self.actor_critic.train()
        
        # Échantillonner un batch du replay buffer
        obs_batch, action_batch, reward_batch, policy_batch, value_batch, continue_batch = \
            self.replay_buffer.sample_batch(self.config.batch_size)
        
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        policy_batch = policy_batch.to(self.device)
        value_batch = value_batch.to(self.device)
        continue_batch = continue_batch.to(self.device)
        
        # Encoder les observations
        batch_size, n_steps = obs_batch.shape[:2]
        obs_flat = obs_batch.reshape(-1, *obs_batch.shape[2:])
        embeddings = self.jepa(obs_flat)  # (batch*n_steps, embed_dim)
        embeddings = embeddings.reshape(batch_size, n_steps, -1)
        
        # Dépiler le World Model sur n_steps
        stoch = torch.zeros(
            batch_size, self.config.world_model.stoch_size,
            self.config.world_model.stoch_classes, device=self.device
        )
        deter = torch.zeros(
            batch_size, self.config.world_model.deter_size,
            device=self.device
        )
        
        wm_loss = 0.0
        for t in range(n_steps):
            post, prior, deter, _ = self.world_model.transition.forward(
                prev_stoch=stoch,
                prev_action=torch.nn.functional.one_hot(
                    action_batch[:, t], self.config.action_space_size
                ).float().to(self.device),
                prev_embedding=embeddings[:, t]
            )
            stoch = post
        
        # Entraînement Actor-Critic sur des rêves
        ac_loss, ac_metrics = self.actor_critic.compute_loss(
            stoch=stoch.unsqueeze(1),
            deter=deter.unsqueeze(1),
            actions=torch.nn.functional.one_hot(
                action_batch[:, -1], self.config.action_space_size
            ).float().to(self.device),
            rewards=reward_batch[:, -1:],
            continues=continue_batch[:, -1:]
        )
        
        total_loss = wm_loss + ac_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.jepa.parameters()) +
            list(self.world_model.parameters()) +
            list(self.actor_critic.parameters()),
            1.0
        )
        self.optimizer.step()
        
        return {
            'wm_loss': wm_loss if isinstance(wm_loss, float) else wm_loss.item(),
            'actor_loss': ac_metrics.get('actor_loss', 0),
            'critic_loss': ac_metrics.get('critic_loss', 0),
            'entropy': ac_metrics.get('entropy', 0),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def _ingest_live_experience(self) -> None:
        """Intègre les expériences de trading live dans le replay buffer."""
        live_dir = Path('results/live_buffer')
        if not live_dir.exists():
            return
        
        for pkl_file in live_dir.glob('*.pkl'):
            try:
                game = GameHistory.load(str(pkl_file))
                self.replay_buffer.add_game(game)
                pkl_file.unlink()  # Supprimer après ingestion
            except Exception:
                pass
    
    def _log_progress(self, step: int, total: int) -> None:
        """Affiche la progression.
        
        Args:
            step: Pas actuel.
            total: Total des pas.
        """
        progress = step / total * 100
        games = self.replay_buffer.num_games
        steps = len(self.replay_buffer)
        lr = self.optimizer.param_groups[0]['lr']
        
        print(
            f"📊 Step {step}/{total} ({progress:.1f}%) | "
            f"Games: {games} | Steps: {steps} | "
            f"LR: {lr:.2e}"
        )
    
    def _save_checkpoint(
        self,
        is_best: bool = False,
        suffix: str = ''
    ) -> None:
        """Sauvegarde un checkpoint du modèle.
        
        Args:
            is_best: Si c'est le meilleur modèle.
            suffix: Suffixe à ajouter au nom du fichier.
        """
        checkpoint = {
            'step': self.step,
            'jepa': self.jepa.state_dict(),
            'world_model': self.world_model.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_{self.step}{suffix}.pth'
        
        torch.save(checkpoint, path)
        
        if not is_best:
            # Garder seulement les 5 derniers checkpoints
            checkpoints = sorted(
                self.checkpoint_dir.glob('checkpoint_*.pth'),
                key=os.path.getmtime
            )
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()