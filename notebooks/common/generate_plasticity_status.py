import numpy as np
import random
import pandas as pd
from tracetype import *


def define_plas_stat(corticostriatal_plasticity,plasticity_change_trial_num,n_trials):
    
    plasticity_status_df = pd.DataFrame()

    trial_index = np.arange(n_trials)
    
    if isinstance(corticostriatal_plasticity,list):
        plasticity_status = []
        for x in trial_index:
            state = corticostriatal_plasticity[np.digitize(x,plasticity_change_trial_num)-1]
            plasticity_status.append(state)
    elif isinstance(corticostriatal_plasticity,str):
        plasticity_status = np.hstack([corticostriatal_plasticity]*n_trials)
        
    plasticity_status_df["trial_num"] = trial_index
    plasticity_status_df["plasticity_status"] = plasticity_status
    
    return plasticity_status_df


def GenPlasStat(corticostriatal_plasticity,plasticity_change_trial_num,n_trials):
    
    print("begin GenPlasStat")
    #reward_t1, reward_t2
    plasticity_status_df = define_plas_stat(corticostriatal_plasticity, plasticity_change_trial_num,n_trials)
    
    print("plasticity_status_df")
    print(plasticity_status_df)
    
    return plasticity_status_df
    