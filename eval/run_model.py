from utils.test_policy import load_policy_and_env, run_policy

if __name__ == "__main__":
    env, act = load_policy_and_env("../results/2020-12-21_mpo/test/", device="cuda:0")
    run_policy(env, act)
