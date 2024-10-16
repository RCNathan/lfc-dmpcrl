from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator

from lfc_env import LtiSystem


class LfcLstdQLearningAgentCoordinator(LstdQLearningAgentCoordinator):
    """A coordinator for LSTD-Q learning agents - for the Lfc problem. This agent hence handles load changes."""    

    def on_episode_start(self, env: LtiSystem, episode: int, state) -> None:
        if self.centralized_flag:
            # print("flag:", self.centralized_flag)
            self.fixed_parameters["Pl"] = env.unwrapped.load 
        else:
            # print("TODO: implement distributed load")
            for i in range(len(self.agents)):
                # self.agents[i].agent._fixed_pars[f'Pl_{i}'] = env.unwrapped.load[i]
                self.agents[i].agent._fixed_pars['Pl'] = env.unwrapped.load[i]
        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: LtiSystem, episode: int, timestep: int) -> None:
        if self.centralized_flag:
            # print("flag:", self.centralized_flag)
            self.fixed_parameters["Pl"] = env.unwrapped.load
        else:
            # print("TODO: implement laod")
            for i in range(len(self.agents)):
                # self.agents[i].agent._fixed_pars[f'Pl_{i}'] = env.unwrapped.load[i]
                self.agents[i].agent._fixed_pars['Pl'] = env.unwrapped.load[i]
        return super().on_env_step(env, episode, timestep)
