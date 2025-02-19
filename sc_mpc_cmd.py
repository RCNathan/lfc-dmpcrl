from lfc_model import Model
from sc_mpc import evaluate_scmpc


# some constants
model = Model()
t_end = 10  # end-time in seconds 
numSteps = int(t_end / model.ts) # default 1000 steps

print("Running 1 episode, 10 scenarios for both scenarios")

# scenario 1 | no perturbations on A, B, F, but noises on load (Pl)
print("Scenario 1")
evaluate_scmpc(
    numEpisodes=20, 
    numSteps=numSteps, 
    scenario=1, 
    n_scenarios=5, 
    make_plots=False,
    solver="ipopt",
    save_name_info="ipopt"
)

# scenario 2 | additionally, perturbations on A, B, F
print("Scenario 2")
evaluate_scmpc(
    numEpisodes=20, 
    numSteps=numSteps, 
    scenario=2, 
    n_scenarios=5, 
    make_plots=False,
    solver="ipopt",
    save_name_info="ipopt"
)

print("All training complete.")