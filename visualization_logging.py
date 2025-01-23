from vis_large_eps import vis_large_eps

"""
File intended to be used for logging visualization info.
Information about env/model settings and changelogs, along with results for different scenarios.
Hyperparameters are found in the bulk_tests.py file, and plots are saved in the PowerPoint (meeting_sam_6_11.ppt)
"""

# file for logging info on visualization, instead of manually turning it on/off everytime.
# import vis_large_eps here, leave vis_large_eps untouched.

# changed Qs,Qx and other stuff -> to get TD error down for numerical stability
filename = "cent_5ep"  # this one shows really promising results!
filename = "cent_50epTEST"  # holy shit this shit is amazing!
# filename = 'cent_5epTEST'
# filename = 'cent_5epTEST3'
# filename = 'cent_5epTEST4'
filename = "cent_20epTEST4"  # incredible. good stuffs!


# Redoing Qs, Qx again, together with sampling time from 0.01s -> 0.1s, changing bounds and constraints (accordance with literature)
filename = "cent_no_learning_1epTEST5"  # unstable behavior after 20ish seconds, when noise on A,B,F and load.
filename = (
    "cent_no_learning_5epTEST5"  # very sensitive to values of noise for some reason
)
filename = "cent_5epTEST5"

# Distributed time - see following for baselines: no noise on load or on A,B,F
filename = "cent_5epdist_time"
filename = "cent_no_learning_1epdist_time"

# added grc
filename = "cent_no_learning_1ep_grc"
# filename = 'cent_no_learning_1ep_grc_P'

# sanity check
filename = "cent_no_learning_5ep_sanityCheck"
filename = "distr_no_learning_2ep_scenario_0"  # distributed, 2 eps @ 20 steps. returns are identical!

# Scenario 0
filename = "cent_5ep_scenario_0"
filename = "cent_no_learning_1ep_scenario_0"  # centralized, return [460.55373678]
filename = "distr_no_learning_1ep_scenario_0"  # distributed, return [459.15050864]

filename = "cent_10ep_scenario_0"  # learning_rate=1e-11 [1400.36087242 1387.50572943 1130.69684112 1083.77528437  960.88003118  996.20473584 1076.01965776 ...]
filename = "cent_50ep_scenario_0"  # learning_rate=1e-11, eps=0.7 [1312 ... 940.13500934  949.62512106  949.30573528  945.88105117]
filename = "distr_1ep_scenario_0"

# # Scenario 1 - noise on load
# filename = 'cent_no_learning_1ep_scenario_1' # [531.66506515]
# filename = 'cent_5ep_scenario_1' # [559.64513404 551.25032843 647.12568547 453.05172057 454.00744462]


# other stuff in between, mainly dual vars checking
filename = "cent_no_learning_5ep_scenario_0.1"  # WHY IS THERE NON DETERMINISM HAPPENING AGAIN - no noise whatsoever
filename = "cent_no_learning_3ep_scenario_0.1"

filename = "distr_no_learning_1ep_scenario_0.1"

# Scenario 0 | GRC = 1 (basically turned off)
filename = "cent_no_learning_5ep_scenario_0.2"  # 5x [658.71297405], GRC = 1? loads +-0.8
filename = 'cent_10ep_scenario_0.2' # learning for GRC=0.1, loads increased +-1.0

# Scenario 0 | load to +-0.085, GRC at 0.1
filename = "cent_no_learning_1ep_scenario_0.2"
filename = "distr_no_learning_1ep_scenario_0.2" # 28-11: full run complete, return [811.76]

# Scenario 1 | load noise 0.03*uniform
# filename= 'cent_5ep_scenario_1'# [3220.49319484  958.90272301  944.0168765   908.90453459  695.90968997]
# filename= 'cent_no_learning_1ep_scenario_1' # [960.91]

# after command line shenanigans
# filename = r"data\pkls\centlearnmanual_cent_5ep_scenario_1"
filename = r"data from server\batch 2\pkls\tcl3_cent_20ep_scenario_1"
filename = r"data\pkls\centlearnmanual_cent_5ep_scenario_1"
filename = r"data\pkls\addMoreLoadinfo_distr_no_learning_1ep_scenario_1"

# batch 3 partly in; tcn2, tdn2, tcl13-15, tdl16,-19,-23
filename = r"data from server\batch 3\pkls\tcl13_cent_20ep_scenario_1"
filename = r"data from server\batch 3\pkls\tdl16_distr_20ep_scenario_1" 
    # I've noted distributed follows centralized very closely, so learning params can be chosen to be the same.
    # Also: learning has not converged after 20 eps, as seen by the learning-params plot that do not converge.
        # so either learning rate lower (it's 1 now), or more episodes.
filename = r"data from server\batch 3\pkls\tdl19_distr_20ep_scenario_1" # 19: upd-freq=2: agressive
filename = r"data from server\batch 3\pkls\tdl23_distr_20ep_scenario_1" # 23: less smooth buffer: not a whole big diff

# DDPG stuff
filename = r"ddpg\ddpg_env_eval" # shapes: x: (20, 301, 12), u: (20, 300, 3), Pl: (1, 6011, 3), Pl_noise: (1, 6011, 3)
# filename = r"ddpg\ddpg_env_train" # shapes: x: (20, 301, 12), u: (20, 300, 3), Pl: (1, 6011, 3), Pl_noise: (1, 6011, 3)

# filename = r"ddpg\ddpg_lfc_changelr2"
filename = r"ddpg\ddpg_env_evaltest" # (1000, 1001, 12) # 10 x 10 = 100 episodes
# filename = r"ddpg\ddpg_env_traintest" # (1000, 1001, 12) # 1000 episodes

# consulting GPT on hyper params: biggest change: batch-size to 256:
filename = r"ddpg\ddpg_env_evaltest2" # (1000, 1001, 12) # 10 x 10 = 100 episodes
# filename = r"ddpg\ddpg_env_traintest2" # (1000, 1001, 12) # 1000 episodes

# filename = r"ddpg\ddpg_env_evaltest3" # (1000, 1001, 12) # 10 x 10 = 100 episodes
filename = r"ddpg\ddpg_env_traintest3" # (1000, 1001, 12) # 1000 episodes

# filename = r"ddpg\ddpg_env_evaltest6" # (1000, 1001, 12) # 10 x 10 = 100 episodes
filename = r"ddpg\ddpg_env_traintest6" # (1000, 1001, 12) # 1000 episodes

filename = r"ddpg\ddpg_env_traintest8" # 

# Scenario 2 | noise on A, B and F: 1e-1, 1e-2, 1e-2
# while working on DDPG, continuing with noise on A, B and F:
# filename = r"data\pkls\scenario2_cent_no_learning_3ep_scenario_1" # note: title is wrong, should be scenario 2 at the end
# filename = r"data from server\batch 5\pkls\tcl26_cent_20ep_scenario_1" 
# lr's of 1e-13, 14, 15 are too small; TD fluctuates but is identical between the 13, 14, 15 runs: lr too small
# filename = r"data\pkls\tcl29_cent_20ep_scenario_1"
# filename = r"data\pkls\tcl32_cent_20ep_scenario_2" # 32-34 all have a lot of infeasibles
# filename = r"data\pkls\tcl34_cent_50ep_scenario_2" # 33 AND 34: ep 31-50 are infeasible -> maybe factor from 1 to 0.99...?


# filename = r"data\pkls\start_manual_cent_30ep_scenario_2"
# filename = r"data\pkls\start_manual_cent_5ep_scenario_2"

filename = r"data\pkls\tcl35_cent_20ep_scenario_2"
filename = r"data\pkls\tcl39_cent_20ep_scenario_2" # infeasible from ep 14
filename = r"data\pkls\tcl40_cent_20ep_scenario_2" # no infeasibles! 
filename = r"data from server\tcl44_cent_50ep_scenario_2" # exceptional! but infeasilbe from ep 8

# 45 - 48
filename = r"data\pkls\tcl48_cent_50ep_scenario_2" # YO WHAT THIS IS LEGIT! very nice however infeasible after ep 31; change lr?!
filename = r"data\pkls\tcl52_cent_100ep_scenario_2" # 
filename = r"data\pkls\tcl58_cent_50ep_scenario_2" 

# fuck me. I might have implemented the experience replay (and possibly others) wrong? Looking at the code,
# it doens't seem that the default values are overridden..?
# yes, unfortunately, experience replay was not altered. Any tests where I changed those settings are worthless...
filename = r"data\pkls\test1_cent_3ep_scenario_2"
filename = r"data\pkls\tcl53_cent_50ep_scenario_2_expfix" 


# By FAR the best results I've gotten are these two:
filename = r"data from server\tcl44_cent_50ep_scenario_2" # exceptional! but infeasilbe from ep 8
filename = r"data\pkls\tcl48_cent_50ep_scenario_2" # YO WHAT THIS IS LEGIT! very nice however infeasible after ep 31; change lr?!

filename = r"data\pkls\tcl58_cent_50ep_scenario_2" # both 57 and 58 are w/o infs and as good as 58! [!!oh wait pre-bug-fix!!]
filename = r"data\pkls\tcl57_cent_50ep_scenario_2_expfix" # updated 57/58 are not good
# filename = r"data\pkls\tcl48rp_cent_50ep_scenario_2_expfix"  # repeatibilty; huh. its identical to 48 but no infeasibility issues now..
# filename = r"data\pkls\tcl62_cent_50ep_scenario_2_expfix"
filename = r"data\pkls\tcl65_cent_100ep_scenario_2" # config25: 63, 64, 65


# note; checking eval for now since the train has file-size of a GB. its huge.
filename = r"ddpg\ddpg_env_evalddpg1" # not good.
filename = r"ddpg\ddpg_env_evalddpg2" # its alright-ish, especially the train plot (see ppt)
filename = r"ddpg\ddpg_env_trainddpg4" # the promised one - eval peaks halfway (really good)


# # scenario 2.1 ; lowered noise values
# filename = r"data\pkls\tcl32_cent_20ep_scenario_2.1" # 26-32: 32 is the best, though infeasible.
# filename = r"data\pkls\tcl34_cent_50ep_scenario_2.1" # config12; 33 decent; violates tie line flow, similar to other exp. 34 is shit.
# # filename = r"data\pkls\tcl40_cent_20ep_scenario_2.1" # config13; 35-39 were manually done, so only 40 exists.
# filename = r"data\pkls\tcl42_cent_50ep_scenario_2.1" # config14: 41, 42: very shitty
# filename = r"data\pkls\tcl43_cent_50ep_scenario_2.1" # config15: 43, 44 both shit
# filename = r"data\pkls\tcl46_cent_50ep_scenario_2.1" # config16: 45, 46 are shite
# filename = r"data\pkls\tcl62_cent_50ep_scenario_2.1" # config24: 48rp, 61, 62



# # noticed a bug in the env; testing now how much it affects training/learning: bug in grc penalization where ts was mistakenly left out.
# filename = r"data\pkls\anotherTestForGrcbugfix_cent_5ep_scenario_2.1" # seems very similar to the one prior to bugfix (slide 111 vs 223 in ppt)
# # also testing for tcl43 - only the first X episodes

vis_large_eps(filename)