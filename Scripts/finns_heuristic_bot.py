''' core imports '''
import numpy as np
import matplotlib.pyplot as plt

''' development imports'''
from time import perf_counter
from tqdm import tqdm

''' visualization imports '''
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('mycmap', ['lightgrey', 'white'])
import matplotlib.colors as mcolors
tab10_names = list(mcolors.TABLEAU_COLORS) # create a list of colours

def checkerboard(shape):
    # from https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    return np.indices(shape).sum(axis=0) % 2

class finns_heuristic_bot:
    '''
    this could use a significant refactor
    this provides a lot of methods to pull and remix
    really doesn't use valid_moves
    (valid moves can be determined from board_state and active_box)
    
    it's not very strong. you should be able to beat this fairly easily.
    '''
    
    ''' ------------------ required function ---------------- '''
    
    def __init__(self,name: str = 'Aljoscha') -> None:
        self.name = name
        
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        return tuple(self.heuristic_mini_to_major(board_state = board_dict['board_state'],
                                                  active_box = board_dict['active_box'],
                                                  valid_moves = board_dict['valid_moves']))
    
    
    ''' --------- generally useful bot functions ------------ '''
    
    def _check_line(self, box: np.array) -> bool:
        '''
        box is a (3,3) array
        returns True if a line is found, else returns False '''
        for i in range(3):
            if abs(sum(box[:,i])) == 3: return True # horizontal
            if abs(sum(box[i,:])) == 3: return True # vertical

        # diagonals
        if abs(box.trace()) == 3: return True
        if abs(np.rot90(box).trace()) == 3: return True
        return False

    def _check_line_playerwise(self, box: np.array, player: int = None):
        ''' returns true if the given player has a line in the box, else false
        if no player is given, it checks for whether any player has a line in the box'''
        if player == None:
            return self._check_line(box)
        if player == -1:
            box = box * -1
        box = np.clip(box,0,1)
        return self._check_line(box)
    
    def pull_mini_board(self, board_state: np.array, mini_board_index: tuple) -> np.array:
        ''' extracts a mini board from the 9x9 given the its index'''
        temp = board_state[mini_board_index[0]*3:(mini_board_index[0]+1)*3,
                           mini_board_index[1]*3:(mini_board_index[1]+1)*3]
        return temp

    def get_valid(self, mini_board: np.array) -> np.array:
        ''' gets valid moves in the miniboard'''
        return np.where(mini_board == 0)

    
    def get_finished(self, board_state: np.array) -> np.array:
        ''' calculates the completed boxes'''
        opp_boxes = np.zeros((3,3))
        self_boxes = np.zeros((3,3))
        stale_boxes = np.zeros((3,3))
        # look at each miniboard separately
        for _r in range(3):
            for _c in range(3):
                temp_miniboard = self.pull_mini_board(board_state, (_r,_c))
                self_boxes[_r,_c] = self._check_line_playerwise(temp_miniboard, player = 1)
                opp_boxes[_r,_c] = self._check_line_playerwise(temp_miniboard, player = -1)
                if sum(abs(temp_miniboard.flatten())) == 9:
                    stale_boxes[_r,_c] = 1                   

        # return finished boxes (separated by their content)
        return (opp_boxes*-1, self_boxes, stale_boxes)
    
    def convert_pos_to_int(self, position: tuple) -> int:
        ''' converts a tuple to a unique location on the board represented by an integer
        (2,4) -> 18 + 4 -> 22 '''
        # comparing tuples is irritating, comparing integers is much easier
        return position[0] * 9 + position[1]
    
    def convert_pos_to_int(self, position: tuple) -> int:
        '''
        currently unused
        
        converts a tuple to a unique location on the board represented by an integer
        (2,4) -> 18 + 4 -> 22 '''
        # comparing tuples is irritating, comparing integers is much easier
        return position[0] * 9 + position[1]
    
    def block_imminent(self, mini_board: np.array) -> list:
        ''' tries to block the opponent if they have 2/3rds of a line '''
        # loop through valid moves with enemy position there.
        # if it makes a line it's imminent
        imminent = list()

        for _valid in zip(*self.get_valid(mini_board)):
            # create temp valid pattern
            valid_filter = np.zeros((3,3))
            valid_filter[_valid[0],_valid[1]] = -1
            if self._check_line(mini_board + valid_filter):
                imminent.append(_valid)
        return imminent
    
    
    ''' ------------------ bot specific logic ---------------- '''
    
    def heuristic_mini_to_major(self, board_state: np.array,
                                active_box: tuple,
                                valid_moves: list) -> tuple:
        '''
        either applies the heuristic to the mini-board or selects a mini-board (then applies the heuristic to it)
        '''

        if active_box != (-1,-1):
            # look just at the mini board
            temp_miniboard = self.pull_mini_board(board_state, active_box)
            # look using the logic, select a move
            move = self.mid_heuristic(temp_miniboard)
            # project back to original board space
            return (move[0] + 3 * active_box[0],
                    move[1] + 3 * active_box[1])

        else:
            # use heuristic on finished boxes to select which box to play in
            imposed_active_box = self.major_heuristic(board_state)

            # call this function with the self-imposed active box
            return self.heuristic_mini_to_major(board_state = board_state,
                                           active_box = imposed_active_box,
                                           valid_moves = valid_moves)

    def major_heuristic(self, board_state: np.array) -> tuple:
        '''
        determines which miniboard to play on
        note: having stale boxes was causing issues where the logic wanted to block
              the opponent but that mini-board was already finished (it was stale)
        '''
        z = self.get_finished(board_state)
        # finished boxes is a tuple of 3 masks: self, opponent, stale 
        self_boxes  = z[0]
        opp_boxes   = z[1]
        stale_boxes = z[2]
        
        # identify imminent wins
        imminent_wins = self.block_imminent(self_boxes + opp_boxes)
        
        # make new list to remove imminent wins that point to stale boxes
        stale_boxes = list(zip(*np.where(stale_boxes)))
        for stale_box in stale_boxes:
            if stale_box in imminent_wins:
                imminent_wins.remove(stale_box)
        if len(imminent_wins) > 0:
            return imminent_wins[np.random.choice(len(imminent_wins))]

        # take center if available
        internal_valid = list(zip(*self.get_valid(self_boxes + opp_boxes)))
        for stale_box in stale_boxes:
            if stale_box in internal_valid:
                internal_valid.remove(stale_box)

        if (1,1) in internal_valid:
            return (1,1)

        # else take random corner
        _corners = [(0,0),(0,2),(0,2),(2,2)]
        _valid_corner = list()

        for _corner in _corners:
            if _corner in internal_valid:
                _valid_corner.append(_corner)
        if len(_valid_corner) > 0:
            return _valid_corner[np.random.choice(len(_valid_corner))]

        # else take random
        return internal_valid[np.random.choice(len(internal_valid))]
        
    def mid_heuristic(self, miniboard: np.array) -> tuple:
        ''' main mini-board logic '''
        # block imminent wins on this miniboard
        imminent_wins = self.block_imminent(miniboard)
        if len(imminent_wins) > 0:
            return imminent_wins[np.random.choice(len(imminent_wins))]

        # take center if available
        internal_valid = list(zip(*self.get_valid(miniboard)))
        if (1,1) in internal_valid:
            return (1,1)

        # else take random corner
        _corners = [(0,0),(0,2),(0,2),(2,2)]
        _valid_corner = list()

        for _corner in _corners:
            if _corner in internal_valid:
                _valid_corner.append(_corner)
        if len(_valid_corner) > 0:
            return _valid_corner[np.random.choice(len(_valid_corner))] # must convert back to full board tuple

        # else take random
        return internal_valid[np.random.choice(len(internal_valid))]
