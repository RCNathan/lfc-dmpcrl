### Sanity Checks ###

# Noise seed affects results a lot; weird.
# centralized, no learning: check if no noise on load, but WITH noise on A,B,F leads to same result every time (if not: problem.)
# train(centralized_flag=True, learning_flag=False, numEpisodes=5, numSteps=300, prediction_horizon=15)
# 1) no stochasticities (none on A,B,F, load): different costs :/
# cost for noise on A,B,F only?
# 2) for noise on A,B,F -> cent_no_learning_5ep_sanityCheck: different costs. Similar? yes.
# 3) for noise on load


# added terminal penalty to cost in learnable mpc.
# train(centralized_flag=True, learning_flag=False, numEpisodes=1, numSteps=500, prediction_horizon=15)


# distributed, no learning: check if same results as centralized -> should be.
# admm_iters = 1000 # tune this and rho if the MPCs are correct but different results.
# rho = 0.1

# After adding GRC, everything breaks. ehm.
# With perfect knowledge (no noise on A,B,F, load), x0 = zeros, and only small step of 0.01, infeasibilities all over the place...
# train(centralized_flag=True, learning_flag=False, numEpisodes=1, numSteps=500, prediction_horizon=15)
# increasing prediction horizon fixed this! in literature: N = 12, 13 and even 15 (Venkat) -> slacking the GRC is necessary!
# filename cent_no_learning_5ep_grc

# fixed with learning?
# train(centralized_flag=True, learning_flag=True, numEpisodes=5, numSteps=200, prediction_horizon=15)
# no xD

# now that GRC gets slacked, horizon can go back to 10

# investigating why stochasticity-free leads to different results still
# [5.47128098e-05] run 1
# [5.47128098e-05] run 2
# [5.47128098e-05 2.07741712e-05 7.27197147e-06] run 3, 3 eps -> clearly its with the reset that introduces differences.
# train(centralized_flag=True, learning_flag=False, numEpisodes=3, numSteps=10, prediction_horizon=10)
