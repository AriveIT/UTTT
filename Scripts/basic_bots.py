import numpy as np

class random_bot:
    '''
    this bot selects a random valid move
    '''
    def __init__(self, name = 'beep-boop'):
        self.name = name
    def move(self, board_dict):
        random_index = np.random.choice(len(board_dict['valid_moves']))
        return board_dict['valid_moves'][random_index]
    
class same_box_bot:
    '''
    this bot tries to play in the same mini board that it currently is in
    it will often return an invalid move, which is converted to a random move.
    '''
    def __init__(self, name = 'stuart'):
        self.name = name
        
    def move(self, board_dict):
        return board_dict['active_box']

class first_bot:
    '''
    the board_dict provides a list of valid moves
    this bot plays the first entry of that list
    '''
    def __init__(self, name = 'dozer'):
        self.name = name
        
    def move(self, board_dict):
        return board_dict['valid_moves'][0]