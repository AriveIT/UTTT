import numpy as np
import time
from functools import lru_cache

# https://github.com/uvicaiclub/UTTT/tree/main

class agent():
    
    # Local coordinates: coordinates for a mini board: Range = (0,0) --> (2,2)
    # Global coordinates: coordinates for complete board: Range = (0,0) --> (8,8)

    ACTIVE = 0
    MAN_BIASES = np.array([[0.6, 0.25, 0.6],
                           [0.25, 1, 0.25],
                           [0.6, 0.25, 0.6]])
    TIME_LIMIT = 0.5 * 0.95

    def __init__(self, name: str = 'ari\'s super cool good bot'):
        self.name = name
        self.best_move_table = {}

        self.evaluation_over_time = []
        self.depth_over_time = []
        self.num_states_over_time = []
    
    def move(self, board_dict: dict) -> tuple:

        start_time = time.perf_counter()

        board_state = board_dict["board_state"]
        active_box = board_dict["active_box"]
        #valid_moves = board_dict["valid_moves"]
        
        won_boxes = self.initialize_won_boxes(board_state) # 1 = won board, -1 = lost board, 0 = active / not won board
        counts = self.initialize_counts(board_state)
        man_advantages = self.initialize_man_advantages(board_state) # keep track of how many X / O are in each box

        """ Iterative deepening """
        cur_depth = 3
        best_move = None
        self.states_evaluated = 0
        self.tt_time = 0
        while True:
            move, eval = self.minimax(board_state, active_box, won_boxes, counts, man_advantages, start_time, cur_depth, float("-inf"), float("inf"), True)

            best_move = move
            best_eval = eval

            if time.perf_counter() - start_time > self.TIME_LIMIT:
                break

            cur_depth += 1

        """ Stats """
        #print(f"total time: {time.perf_counter() - start_time}")
        # if best_eval == float("inf"): self.evaluation_over_time.append(100)
        # elif best_eval == float("-inf"): self.evaluation_over_time.append(-100)
        # else: self.evaluation_over_time.append(best_eval)
        # self.depth_over_time.append(cur_depth - 1)
        # self.num_states_over_time.append(self.states_evaluated)

        return best_move

    """
            Logic
    """
    def minimax(self, board_state, active_box, won_boxes, counts, man_advantages, start_time, depth, alpha, beta, maximizingPlayer):
        self.states_evaluated += 1

        """ Base Cases """

        # if game won
        won = self.check_line(won_boxes)
        if won != 0:
            
            return (-1,-1), float("inf") if won == 1 else float("-inf")

        # if tie
        full_boxes_mask = self.get_full_boxes_mask(counts)
        if np.count_nonzero(won_boxes + full_boxes_mask) == 9:
            return (-1,-1), 0

        if depth == 0:
            evaluation = self.evaluate_position(won_boxes, man_advantages, board_state)
            return (-1,-1), evaluation
        

        """ Minimax """
        # order moves
        possible_moves = self.get_valid_moves(active_box, board_state, counts, won_boxes)
        self.move_ordering(possible_moves, board_state)

        if maximizingPlayer:

            maxEval = float("-inf")
            bestMove = possible_moves[0]

            for move in possible_moves:
            
                self.take_turn(1, move, board_state, counts, man_advantages, won_boxes)

                # continue down tree
                _, evaluation = self.minimax(board_state, self.get_next_active_box(move, won_boxes, counts), won_boxes, counts, man_advantages, start_time, depth-1, alpha, beta, False)

                #if out of time
                if time.perf_counter() - start_time > self.TIME_LIMIT:
                    return bestMove, maxEval
                
                self.undo_turn(1, move, board_state, counts, man_advantages, won_boxes)

                if evaluation > maxEval:
                    maxEval = evaluation
                    bestMove = move
                
                # alpha beta pruning
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break

            self.set_best_move(board_state, bestMove)
            return bestMove, maxEval
        else:

            minEval = float("inf")
            bestMove = possible_moves[0]

            for move in possible_moves:
                # if out of time
                if time.perf_counter() - start_time > self.TIME_LIMIT:
                    return bestMove, minEval
                
                self.take_turn(-1, move, board_state, counts, man_advantages, won_boxes)

                # continue down tree
                _, evaluation = self.minimax(board_state, self.get_next_active_box(move, won_boxes, counts), won_boxes, counts, man_advantages, start_time, depth-1, alpha, beta, True)

                self.undo_turn(-1, move, board_state, counts, man_advantages, won_boxes)

                if evaluation < minEval:
                    minEval = evaluation
                    bestMove = move

                # alpha beta pruning
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break

            self.set_best_move(board_state, bestMove)
            return bestMove, minEval

    def take_turn(self, val, move, board_state, counts, man_advantages, won_boxes):
        box_coord = self.get_box_for_coord(move)

        board_state[move[0],move[1]] = val
        man_advantages[box_coord] += val
        self.update_won_boxes(board_state, box_coord, won_boxes)
        counts[box_coord] += 1


    def undo_turn(self, val, move, board_state, counts, man_advantages, won_boxes):
        box_coord = self.get_box_for_coord(move)

        board_state[move[0],move[1]] = 0
        man_advantages[box_coord] -= val
        won_boxes[self.get_box_for_coord(move)] = 0
        counts[box_coord] -= 1

    """
            Transposition Table
    """
    def set_best_move(self, board_state, best_move):
        start_time = time.perf_counter()

        id = hash(board_state.data.tobytes())
        self.best_move_table[id] = best_move

        self.tt_time += (time.perf_counter() - start_time)
    
    def get_best_move(self, board_state):
        start_time = time.perf_counter()

        id = hash(board_state.data.tobytes())

        self.tt_time += (time.perf_counter() - start_time)
        return self.best_move_table.get(id)
    
        
   
    """
            Move ordering
    """
    def move_ordering(self, moves, board_state):
        start_time = time.perf_counter()
        best_move = self.get_best_move(board_state)
        
        if best_move is None:
            return

        for i in range(len(moves)):
            if moves[i][0] == best_move[0] and moves[i][1] == best_move[1]:
                moves[0], moves[i] = moves[i], moves[0]

        self.tt_time += (time.perf_counter() - start_time)

        return
 
    """ 
            Position Evaluation
    """
    def evaluate_position(self, won_boxes, man_advantages, board_state):
        # kinda random values atm
        won_boxes_multiplier = 2 
        two_in_a_row_multiplier = 0 # not used (currently hurts performance)
        two_in_a_row_bonus_multiplier = 5
        man_multiplier = 1 # no clue if this helps at all

        # count won squares
        won_boxes_sum = sum(sum(won_boxes))

        # bonus for won squares with 2 in a row
        two_in_a_row_bonus = self.count_two_in_a_row_in_a_box(won_boxes)

        # count 2 in a row
        two_in_a_row_sum = 0 #self.count_two_in_a_row(board_state, won_boxes)

        # man advantages
        man_score = self.get_man_advantages_score(man_advantages)

        evaluation = won_boxes_sum * won_boxes_multiplier + \
                        two_in_a_row_sum * two_in_a_row_multiplier + \
                        two_in_a_row_bonus * two_in_a_row_bonus_multiplier + \
                        man_score * man_multiplier

        return evaluation

    def get_man_advantages_score(self, man_advantages):
        temp = np.zeros((3,3))
        for x in range(3):
            for y in range(3):
                if man_advantages[x,y] > 0: temp[x,y] = 1
                elif man_advantages[x,y] < 0: temp[x,y] = -1
                else: temp[x,y] = 0
        return np.sum(temp * self.MAN_BIASES)

    # counts how many 2 in a rows there are that aren't in won boxes
    def count_two_in_a_row(self, board_state, won_boxes):
        two_in_a_row_sum = 0
        for i in range(9):
            x = i % 3
            y = i // 3
            if won_boxes[x,y] != 0: continue

            box = self.pull_mini_board(board_state, (x, y))
            
            two_in_a_row_sum += self.count_two_in_a_row_in_a_box(box)
        return two_in_a_row_sum
    
    def count_two_in_a_row_in_a_box(self, box):
        two_in_a_row_sum = 0

        for i in range(3):
            temp = sum(box[:,i]) # horizontal
            if abs(temp) == 2: two_in_a_row_sum += temp

            temp = sum(box[i,:]) # vertical
            if abs(temp) == 2: two_in_a_row_sum += temp

        # diagonals
        temp = box.trace()
        if abs(temp) == 2: two_in_a_row_sum += temp

        temp = np.rot90(box).trace()
        if abs(temp) == 2: two_in_a_row_sum += temp
        
        return two_in_a_row_sum


    """
            General Utility
    """
    def get_full_boxes_mask(self, counts):
        full_boxes_mask = np.zeros((3,3))
        for x in range(3):
            for y in range(3):
                if counts[x,y] == 9:
                    full_boxes_mask[x,y] = 10 # 10, so 10 + -1 != 0
        return full_boxes_mask

    def initialize_won_boxes(self, board_state):
        won_boxes = np.zeros((3,3))
        for x in range(3):
            for y in range(3):
                box = self.pull_mini_board(board_state, (x,y))
                won_boxes[x,y] = self.check_line(box)

        return won_boxes

    def update_won_boxes(self, board_state, box_coord, won_boxes):
        box = self.pull_mini_board(board_state, box_coord)
        won_boxes[box_coord] = self.check_line(box)

    def initialize_counts(self, board_state):
        counts = np.zeros((3,3))
        for x in range(3):
            for y in range(3):
                counts[x,y] = np.count_nonzero(board_state[x*3:(x+1)*3, y*3:(y+1)*3])
        return counts

    def update_counts(self, move, counts):
        box = self.get_box_for_coord(move)
        counts[box] += 1

    def initialize_man_advantages(self, board_state):
        man_advantages = np.zeros((3,3))
        for x in range(3):
            for y in range(3):
                man_advantages[x,y] = np.sum(self.pull_mini_board(board_state, (x,y)))
        return man_advantages

    def check_line(self, box: np.array) -> bool:
        '''
        box is a (3,3) array (typically a mini-board)
        returns 1 of -1 if a line is found
        '''
        for i in range(3):
            temp = sum(box[:,i])
            if temp == 3: return 1
            elif temp == -3: return -1 # horizontal
            
            temp = sum(box[i,:])
            if temp == 3: return 1 # vertical
            elif temp == -3: return -1

        # diagonals
        temp = box.trace()
        if temp == 3: return 1
        elif temp == -3: return -1

        temp = np.rot90(box).trace()
        if temp == 3: return 1
        elif temp == -3: return -1

        # if no lines found
        return 0

    def get_valid_moves(self, active_box, board_state, counts, won_boxes):
        '''
        returns valid moves for current board_state and active_box in global coordinates
        '''
        moves = []

        if active_box == (-1,-1):
            for x in range(3):
                for y in range(3):
                    if won_boxes[x,y] == self.ACTIVE and counts[x,y] < 9: # still not won
                        temp = self.get_valid_in_mini(self.pull_mini_board(board_state, (x,y)))
                        temp = self.local_to_global_coords(temp, (x,y))

                        if len(moves) == 0:
                            moves = temp
                        else:
                            moves = np.concatenate((moves, temp))
        else:
            moves = self.get_valid_in_mini(self.pull_mini_board(board_state, active_box))

            # change moves to global coordinates
            moves = [(move[0] + 3 * active_box[0], move[1] + 3 * active_box[1]) for move in moves]

        return moves

    def get_valid_in_all(self, board_state):
        valid = []
        for coord in zip(*np.where(board_state == 0)):
            valid.append(coord)
        return valid

    def get_valid_in_mini(self, mini_board: np.array) -> np.array:
        ''' gets valid moves in the miniboard, in local coords'''
        valid = []
        for coord in zip(*np.where(mini_board == 0)):
            valid.append(coord)
        return valid

    def pull_mini_board(self, board_state: np.array, mini_board_index: tuple) -> np.array:
        ''' extracts a mini board from the 9x9 given its index'''
        return board_state[mini_board_index[0]*3 : (mini_board_index[0]+1)*3, mini_board_index[1]*3 : (mini_board_index[1]+1)*3]
    
    def local_to_global_coords(self, coords, box):
        return [(coord[0] + 3 * box[0], coord[1] + 3 * box[1]) for coord in coords]
    
    def global_to_local_coords(self, coords):
        return (coords[0] % 3, coords[1] % 3)
    
    def get_box_for_coord(self, coord):
        return (coord[0] // 3, coord[1] // 3)
    
    def get_next_active_box(self, last_move, won_boxes, counts):
        box = self.global_to_local_coords(last_move)
        if won_boxes[box] != self.ACTIVE: return (-1,-1)
        if counts[box] == 9: return (-1,-1)
        return box

