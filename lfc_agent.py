from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator

from lfc_env import LtiSystem


class LfcLstdQLearningAgentCoordinator(LstdQLearningAgentCoordinator):
    """A coordinator for LSTD-Q learning agents - for the Lfc problem. This agent hence handles load changes."""    

    def on_episode_start(self, env: LtiSystem, episode: int, state) -> None:
        if self.centralized_flag:
            self.fixed_parameters["Pl"] = env.unwrapped.load 
        else:
            for i in range(len(self.agents)):
                self.agents[i].fixed_parameters['Pl'] = env.unwrapped.load[i]
        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: LtiSystem, episode: int, timestep: int) -> None:
        if self.centralized_flag:
            self.fixed_parameters["Pl"] = env.unwrapped.load
        else:
            for i in range(len(self.agents)):
                self.agents[i].fixed_parameters['Pl'] = env.unwrapped.load[i]
        return super().on_env_step(env, episode, timestep)
    
    # numInfeasibles 
    numInfeasibles = {} # make empty dict
    def on_mpc_failure(self, episode: int, timestep: int | None, status: str, raises: bool) -> None:
        if episode in self.numInfeasibles:
            self.numInfeasibles[episode].append(timestep)
        else:
            self.numInfeasibles[episode] = [timestep]
        return super().on_mpc_failure(episode, timestep, status, raises)
