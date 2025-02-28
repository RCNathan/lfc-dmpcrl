from lfc_model import Model
from sc_mpc import evaluate_scmpc


# some constants
# model = Model()
t_end = 10  # end-time in seconds 
numSteps = int(t_end / Model.ts) # default 1000 steps

# print("Running 20 episodes, 10 scenarios for both scenarios")

# scenario 0 | no perturbations on A, B, F (env no noise on loads)
print("Scenario 0")
evaluate_scmpc(
    numEpisodes=2,
    numSteps=numSteps, 
    scenario=0, # scenario 0 forces n_scenarios = 1!
    n_scenarios=10, # artificial; will be 1.
    make_plots=False,
    solver="ipopt",
    save_name_info="ipopt"
)
evaluate_scmpc(
    numEpisodes=2,
    numSteps=numSteps, 
    scenario=0, # scenario 0 forces n_scenarios = 1!
    n_scenarios=5, # artificial; will be 1.
    make_plots=False,
    solver="ipopt",
    save_name_info="ipopt"
)

# # scenario 1 | no perturbations on A, B, F, but noises on load (Pl)
# print("Scenario 1")
# evaluate_scmpc(
#     numEpisodes=20, 
#     numSteps=numSteps, 
#     scenario=1, 
#     n_scenarios=5, 
#     make_plots=False,
#     solver="ipopt",
#     save_name_info="ipopt"
# )

# # scenario 2 | additionally, perturbations on A, B, F
# print("Scenario 2")
# evaluate_scmpc(
#     numEpisodes=20, 
#     numSteps=numSteps, 
#     scenario=2, 
#     n_scenarios=5, 
#     make_plots=False,
#     solver="ipopt",
#     save_name_info="ipopt"
# )

print("All training complete.")