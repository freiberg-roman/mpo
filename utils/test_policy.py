"""
All functions are heavily inspired by Open AI's implementation but modified for
PyTorch only use case. Documentation is omitted since it adds no additional
information to original documentation
"""
import os
import os.path as osp
import torch
import joblib
import time


def load_policy_and_env(fpath, device="cpu"):
    itr = "last"
    # will always load last policy from save, along with RL env.
    pytsave_path = osp.join(fpath, 'pyt_save')
    # Each file in this folder has naming convention 'modelXX.pt', where
    # 'XX' is either an integer or empty string. Empty string case
    # corresponds to len(x)==8, hence that case is excluded.
    saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]

    itr = '%d' % max(saves) if len(saves) > 0 else ''

    # get action function
    get_action = load_pytorch_policy(fpath, itr, device)
    try:
        state = joblib.load(osp.join(fpath, "vars" + itr + ".pkl"))
        env = state["env"]
    except:
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, device):
    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=10, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    from utils.logx import EpochLogger
    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()