## Reinforcement learning on Hockey with self-play

### Basic training

Soft actor critic:
```
python sac/main.py
```
Twin delayed DDPG:
```
python td3/td3.py
```

The checkpoints will be saved inside `agents/`. There are already multiple checkpoints inside `agents/checkpoints`. Feel free to use them as opponents for self-play (see later).

## Evaluating agents

```
python eval.py
```

Look at them play against each other:

```
agent1_path = "agents/sac_agent.pt"
agent2_path = "agents/td3_agent.pt"

player1 = load_agent(agent1_path)
player2 = load_agent(agent2_path)

play_hockey(player2, player1, num_episodes=100, render=True)
```

Or evaluate all agents:

```
evaluate_all_agents("agents_dir="agents/", num_episodes=1000)
```

## Self-play

First, create a directory and put some agents (call them "opponents") inside. Then, choose a specific agent (call it "player") to train it against these other agents. Periodically, a version of the player will be saved to the directory, and appended to the list of opponents.

The opponents are chosen at random every episode, and stronger opponents are chosen more frequently.

```
python sac_self_play.py --initial_checkpoint <path/to/your/agent.pth> --checkpoint_dir <dir/containing/agents>
```

## Weights and Biases

Metrics will be logged to Weights and Biases: https://wandb.ai/

First, create an account, get an API key, and login:

```
wandb.login(key=xxx, verify=True)
```

Then, just run any script above, and you should see metrics appear on the dashboard.