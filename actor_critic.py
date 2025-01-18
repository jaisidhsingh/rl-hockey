import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, intermediate_dims, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dims = intermediate_dims
        self.output_dim = output_dim
        self.num_layers = len(intermediate_dims) + 1

        self.layers = []
        current_dim = input_dim
        next_dims = intermediate_dims + [output_dim]

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(current_dim, next_dims[i]))
            current_dim = next_dims[i]

            if i != self.num_layers - 1:
                self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, args, env):
        super().__init__()
        hidden_dims = [int(x) for x in args.model_hidden_dims.split(",")]
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]

        self.policy = MLP(self.state_dim, hidden_dims, self.action_dim)
        self.Q_function_1 = MLP(self.state_dim + self.action_dim, hidden_dims, 1)
        self.Q_function_2 = MLP(self.state_dim + self.action_dim, hidden_dims, 1)
        self.device = args.device
    
    def forward(self, state):
        # if type(state) is not torch.Tensor:
        #     state = torch.from_numpy(state).to(self.device)
        return self.action_limit * self.policy(state)
    
    def Q1(self, state, action):
        return self.Q_function_1(torch.cat([state, action], dim=1).to(state.device))
    
    def Q2(self, state, action):
        return self.Q_function_2(torch.cat([state, action], dim=1).to(state.device))
