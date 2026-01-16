import logging

import coloredlogs
from Coach import Coach
from lkid.LKIDGame import LKIDGame
from lkid.keras.NNet import NNetWrapper as nn
from utils import *

Game = LKIDGame

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 2,
    'numEps': 10,
    'tempThreshold': 3,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 20000,
    'numMCTSSims': 5,
    'arenaCompare': 4,
    'cpuct': 1,
    'shaping_weight': 0.2,      # Reward shaping weight
    'church_weight': 0.8,       # Church adjacency weight

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 40,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
