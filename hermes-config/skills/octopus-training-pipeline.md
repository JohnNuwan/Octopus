# Octopus Training Pipeline

Lance un entraînement complet du modèle RL Octopus sur GPU.

## Étapes

1. **Activer l'environnement virtuel**
   ```
   source /home/kadeva/trading-env/bin/activate
   ```

2. **Naviguer dans le moteur**
   ```
   cd /home/kadeva/Octopus/engine
   ```

3. **Vérifier la disponibilité du GPU**
   ```
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```
   Si `False`, abandonner et alerter l'utilisateur.

4. **Lancer l'entraînement**
   ```
   timeout 36000 python3 -m src.training --steps 30000
   ```
   Timeout après 10h pour éviter les processus zombies.

5. **Analyser les résultats**
   ```
   python3 -c "
   import torch, glob
   ckpts = sorted(glob.glob('weights/checkpoint*.pth'))
   if ckpts:
       ckpt = torch.load(ckpts[-1])
       print('Steps:', ckpt.get('step'))
       print('Reward:', ckpt.get('total_reward'))
       print('Loss:', ckpt.get('loss'))
   else:
       print('Aucun checkpoint trouvé')
   "
   ```

6. **Envoyer le rapport**
   - Résumé des métriques (reward, loss, steps)
   - Graphiques tensorboard (si disponibles)
   - Statut GPU après entraînement
   - Durée totale d'exécution

## Sortie attendue

Rapport formaté contenant :
```
Training Status: ✅ Success
Steps completed: 30000/30000
Final reward: 1.47
Time elapsed: 8h 23m
GPU util: 92%
```