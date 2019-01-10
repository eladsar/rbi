from config import consts, args
from logger import logger
from experiment import Experiment
import torch
import time


def main():

    torch.set_num_threads(1000)
    print("Torch %d" % torch.get_num_threads())
    # print args of current run
    logger.info("Welcome to Learning from Demonstration simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    with Experiment(logger.filename) as exp:

        if args.learn:
            logger.info("Enter RBI Learning Session, it might take a while")
            exp.learn()

        elif args.play:
            logger.info("Enter RBI playing Session, I hope it goes well")
            exp.play()

        elif args.choose:
            logger.info("Choosing best player")
            exp.choose()

        elif args.multiplay:
            logger.info("Start a multiplay Session")
            exp.multiplay()

        elif args.clean:
            logger.info("Clean old trajectories")
            exp.clean()

        else:
            raise NotImplementedError

    logger.info("End of simulation")


if __name__ == '__main__':
    main()

