from tqdm import tqdm
from dataset import get_dataloader
from config import get_config
import numpy as np
from agent import get_agent



def test():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create dataloader
    test_loader = get_dataloader('test', config)
    config.max_iters = None

    # create network and training agent
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)


    # test
    test_loss = np.array([])
    test_err_deg = np.array([])
    testbar = tqdm(test_loader)
    for i, data in enumerate(testbar):
        pred, loss, err_deg = agent.val_func(data)
        test_loss = np.append(test_loss, loss.item())
        test_err_deg = np.append(test_err_deg, err_deg.detach().cpu().numpy())

    print(f'==== exp: {config.exp_name} ====')
    print(f'10acc: {np.sum(test_err_deg<10)/len(test_err_deg):.3f}')
    print(f'15acc: {np.sum(test_err_deg<15)/len(test_err_deg):.3f}')
    print(f'20acc: {np.sum(test_err_deg<20)/len(test_err_deg):.3f}')
    print(f'mean: {np.mean(test_err_deg):.2f}')
    print(f'median: {np.median(test_err_deg):.2f}')
    print(f'std: {np.std(test_err_deg):.2f}')
    np.save(f'../exps/{config.exp_name}.npy', test_err_deg)



if __name__ == '__main__':
    test()

