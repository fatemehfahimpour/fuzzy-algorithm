from step1.membership_function import trapmf, trimf
import config


def t_in_membership(x):
    return {
        "Cold": trapmf(x, *config.T_COLD_PARAMS),
        "Normal": trimf(x, *config.T_NORMAL_PARAMS),
        "Hot": trapmf(x, *config.T_HOT_PARAMS)
    }


def h_in_membership(x):
    return {
        "Dry": trapmf(x, *config.H_DRY_PARAMS),
        "Normal": trimf(x, *config.H_NORMAL_PARAMS),
        "Wet": trapmf(x, *config.H_WET_PARAMS),
    }


