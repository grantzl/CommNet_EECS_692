from gym.envs.registration import register

register(
    id='PredatorPrey-v0',
    entry_point='ic3net_envs.predator_prey_env_test:PredatorPreyEnv',
)

register(
    id='TrafficJunction-v0',
    entry_point='ic3net_envs.traffic_junction_env:TrafficJunctionEnv',
)
