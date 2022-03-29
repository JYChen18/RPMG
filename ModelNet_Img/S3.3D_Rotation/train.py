from tqdm import tqdm
from dataset import get_dataloader
from config import get_config
import numpy as np
from agent import get_agent


def main():
    # create experiment config containing all hyperparameters
    config = get_config('train')

    # create dataloader
    train_loader = get_dataloader('train', config)
    test_loader = get_dataloader('test', config)
    config.max_iters = config.max_epoch * len(train_loader)

    # create network and training agent
    agent = get_agent(config)

    # load from checkpoint if provided
    if config.cont:
        agent.load_ckpt(config.ckpt)

    # start training
    clock = agent.clock

    for e in range(clock.epoch, config.max_epoch):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            pred, loss, err_deg = agent.train_func(data)     # pred: (b, 9)

            if not config.no_decay:
                lr = agent.adjust_learning_rate_by_epoch(agent.optimizer, clock.epoch, config.max_epoch)
            else:
                lr = config.lr

            if agent.clock.iteration % config.log_frequency == 0:
                agent.writer.add_scalar('train/loss', loss, clock.iteration)
                agent.writer.add_scalar('train/err_mean', err_deg.mean().item(), clock.iteration)
                agent.writer.add_scalar('train/lr', lr, clock.iteration)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix({'loss': loss.item()})

            clock.tick()

        clock.tock()

        if clock.epoch % config.val_frequency == 0:
            test_loss = np.array([])
            test_err_deg = np.array([])
            testbar = tqdm(test_loader)
            for i, data in enumerate(testbar):
                pred, loss, err_deg = agent.val_func(data)
                test_loss = np.append(test_loss, loss.item())
                test_err_deg = np.append(test_err_deg, err_deg.detach().cpu().numpy())

            agent.writer.add_scalar('test/loss', test_loss.mean(), clock.iteration)
            agent.writer.add_scalar('test/err_median', np.median(test_err_deg), clock.iteration)
            agent.writer.add_scalar('test/err_mean', np.mean(test_err_deg), clock.iteration)
            agent.writer.add_scalar('test/err_max', np.max(test_err_deg), clock.iteration)
            agent.writer.add_scalar('test/err_std', np.std(test_err_deg), clock.iteration)
            agent.writer.add_scalar('test/acc_5deg', (test_err_deg < 5).sum() / len(test_err_deg), clock.iteration)
            agent.writer.add_scalar('test/acc_10deg', (test_err_deg < 10).sum() / len(test_err_deg), clock.iteration)
            agent.writer.add_scalar('test/acc_20deg', (test_err_deg < 20).sum() / len(test_err_deg), clock.iteration)
            agent.writer.add_scalar('test/acc_30deg', (test_err_deg < 30).sum() / len(test_err_deg), clock.iteration)

        if clock.epoch % config.save_frequency == 0:
            agent.save_ckpt()
        agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
