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

The checkpoints will be saved inside `agents/`.

## Evaluating agents

```
python eval.py
```

Look at them play against each other:

```
agent1_path = "agents/sac_agent_sp1M.pt"
    agent2_path = "agents/sac_agent_sp1M_mixed.pt"

    player1 = load_agent(agent1_path)
    player2 = load_agent(agent2_path)

    play_hockey(player2, player1, num_episodes=100, render=True)
```

Or evaluate one agent against all other agents:

```
agent_path = "agents/sac_agent_sp1M.pt"
player = load_agent(agent1_path)
test_all_agents(player, opponent_dir="agents", num_episodes=100, uniform=True)
```

The output shows the win ratio of your player against the other agents.

```
Agent 1: sac_agent_sp1M.pt, Avg result: 0.08, Play count: 26.0
Agent 3: td3_agent.pt, Avg result: -0.59, Play count: 37.0
Agent 4: vanilla_sac_agent.pt, Avg result: -0.78, Play count: 27.0
Agent 2: sac_agent_sp1M_mixed.pt, Avg result: -0.88, Play count: 32.0
Agent 0: sac_agent.pt, Avg result: -1.00, Play count: 28.0
```

## Self-play

Currently only SAC self-play is supported. TD3 will be added soon.

First, create a directory and put some agents (call them "opponents") inside. Then, choose a specific agent (call it "player") to train it against these other agents. Periodically, a version of the player will be saved to the directory, and appended to the list of opponents.

The opponents are chosen at random every episode, and stronger opponents are chosen more frequently.

```
python sac_self_play.py --initial_checkpoint <path/to/your/agent.pth> --checkpoint_dir <dir/containing/agents> --n_step_td 3 --prioritized_replay
```

## Weights and Biases

Metrics will be logged to Weights and Biases: https://wandb.ai/

First, create an account, get an API key, and login:

```
wandb.login(key=xxx, verify=True)
```

Then, just run any script above, and you should see metrics appear on the dashboard.