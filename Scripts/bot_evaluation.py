''' core imports '''
import numpy as np
import matplotlib.pyplot as plt
import time

''' development imports'''
from time import perf_counter
from tqdm import tqdm

''' agents '''
from V11 import agent as last_agent
from V12 import agent 
from finns_heuristic_bot import finns_heuristic_bot
from basic_bots import random_bot

''' engine stuff'''
from uttt_engine import uttt_engine, bot_template

''' visualization imports '''
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('mycmap', ['lightgrey', 'white'])
import matplotlib.colors as mcolors
tab10_names = list(mcolors.TABLEAU_COLORS) # create a list of colours

def main():
    games() 
    #dev_tool()

def games():
    agent1 = agent(name = 'V12')
    agent2 = last_agent(name = 'V11')

    stats = run_many_games(agent1 = agent1, agent2 = agent2, n_games = 10)

    print(f'{agent1.name} wins:', sum(stats == 1))
    print(f'{agent2.name} wins:', sum(stats == -1))
    print('draws:', sum(stats == 0))


""" Useful Development Tool """
def dev_tool():
    np.random.seed(123)

    #while True:
    ''' -- initialize a board ... '''
    engine = uttt_engine()
    engine.load_agents(random_bot(name = 'the bot known as player 1'),
                    random_bot(name = 'the second bot'))
    ''' ... and play some moves randomly -- '''
    for i in range(20):
        if engine.finished == False:
            engine.query_player(loud=False)
    #if engine.active_box == (-1,-1): break
    
    ''' -- visualize -- '''
    engine.draw_board(ticks='on')
    engine.draw_valid_moves()


    ''' -- initialize your bot -- '''
    # my_bot = your_bot(name='test version 0.1')
    # my_bot = finns_heuristic_bot(name='example logical bot')
    my_bot = agent(name="me")

    ''' see what your bot thinks is a good move '''
    proposed_move = my_bot.move(engine.get_query_dict())

    ''' visualize the proposed move '''
    plt.scatter(proposed_move[0],proposed_move[1],marker='+',c='k',s=200)
    plt.show()

""" Bot evaluation """
def initialize(engine_instance, n_moves:int) -> None:
    ''' plays some number of random moves to initialize a game '''
    if n_moves%2 != 0:
        print('warning: number of moves should be even!')
    
    for i in range(n_moves):
        valid_moves = engine_instance.get_valid_moves()
        random_index = np.random.choice(len(valid_moves))
        engine_instance.move(tuple(valid_moves[random_index]))

def run_many_games(agent1: bot_template,
                   agent2: bot_template,
                   n_games: int = 1000,
                   n_init_moves: int = 4):
    ''' repeatedly plays games between two bots to evaluate their fraction of wins '''
    # NOTE: this doesn't switch which player goes first. There may be a mild first-player advantage
    
    np.random.seed(123)
    

    wins = list()
    for i in tqdm(range(n_games)):
        finished_flag = False
        engine = uttt_engine()
        engine.load_agents(agent1, agent2)
        initialize(engine, n_moves=n_init_moves)
        while not finished_flag:
            engine.query_player()

            if engine.finished:
                finished_flag = True
        wins.append(engine.getwinner())

        """ Plots """
        if False:
            # Plot evaluation over time
            plt.subplot(311)
            plt.plot(agent1.evaluation_over_time, color="blue")
            plt.plot(agent2.evaluation_over_time, color="red")
            plt.title("Evaluations")
            agent1.evaluation_over_time = []
            agent2.evaluation_over_time = []

            # Plot depth
            plt.subplot(312)
            plt.plot(agent1.depth_over_time, color="blue")
            plt.plot(agent2.depth_over_time, color="red")
            plt.title("Depth")
            agent1.depth_over_time = []
            agent2.depth_over_time = []
            
            # Plot number of states evaluated
            plt.subplot(313)
            plt.plot(agent1.num_states_over_time, color="blue")
            plt.plot(agent2.num_states_over_time, color="red")
            plt.title("States evaluated")
            agent1.num_states_over_time = []
            agent2.num_states_over_time = []

            # print(engine.get_game_log())
            plt.show()

        
    # return stats
    if sum(wins) > 0: print(agent1.name, 'is the overall winner')
    if sum(wins) < 0: print(agent2.name, 'is the overall winner')
    if sum(wins) == 0: print(agent1.name,'and',agent2.name,'are evenly matched')
    return np.array(wins)

if __name__ == "__main__":
    main()