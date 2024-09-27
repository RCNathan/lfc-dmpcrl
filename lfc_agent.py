from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator

from lfc_env import LtiSystem


class LfcLstdQLearningAgentCoordinator(LstdQLearningAgentCoordinator):
    """A coordinator for LSTD-Q learning agents - for the Lfc problem. This agent hence handles load changes."""

    def on_episode_start(self, env: LtiSystem, episode: int, state) -> None:
        self.fixed_parameters["Pl"] = env.unwrapped.load 
        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: LtiSystem, episode: int, timestep: int) -> None:
        self.fixed_parameters["Pl"] = env.unwrapped.load
        return super().on_env_step(env, episode, timestep)
