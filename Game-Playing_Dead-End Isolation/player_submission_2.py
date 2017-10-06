#!/usr/bin/env python
from isolation import Board, game_as_text
from random import choice

#multiple entry check + increased depth, remove 2nd player check

# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.

# Submission Class 1
class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Evaluation function that outputs a score equal to how many 
        moves are open for AI player on the board.
            
        Args
            param1 (Board): The board and game state.
            param2 (bool): True if maximizing player is active.

        Returns:
            float: The current state's score. Number of your agent's moves.
            
        """
#       n_player = game.move_count%2
#       print n_player;        
        #if (maximizing_player_turn==True):
        return len(game.get_legal_moves())
        
        # TODO: finish this function!
        #raise NotImplementedError


# Submission Class 2
class CustomEvalFn:

    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Custom evaluation function that acts however you think it should. This 
        is not required but highly encouraged if you want to build the best 
        AI possible.
        
        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.
            
        """        
        # implement my_moves - 2xmy_opp_moves
        my_moves = 0
        my_opp_moves = 0
        opp_move_list=[]

        legal_moves = game.get_legal_moves()
        my_moves = len(legal_moves)

        #check opponent's moves
        if(my_moves > 0):
            for mov in legal_moves: #can still go deeper
                game_forecast = game.forecast_move(mov)  # forecast to get a new board
                my_opp_moves = len(game_forecast.get_legal_moves())
                opp_move_list.append(my_opp_moves)

            my_opp_moves = max(opp_move_list)  # argmax

        #my_moves - my_opp
        return (my_moves - my_opp_moves)

        # TODO: finish this function!
        #raise NotImplementedError


class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()): #OpenMoveEvalFn
        """Initializes your player.

        if you find yourself with a superior eval function, update the default 
        value of `eval_fn` to `CustomEvalFn()`
        
        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """

        self.backup_time = 500 #default recursion time
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent
        
        Args:
            game (Board): The board and game state.
            legal_moves (dict): Dictionary of legal moves and their outcomes
            time_left (function): Used to determine time left before timeout
            
        Returns:
            (tuple): best_move
        """
        if (3, 3) in legal_moves:
            return (3, 3)

        # if game.move_count > 16:
        #     for mov in legal_moves:
        #         game_forecast = game.forecast_move(mov)  # forecast to get a new board
        #         if len(game_forecast.get_legal_moves()) == 0:
        #             return mov

        # scale backup/recursion time according to size of the board
        self.backup_time = time_left()*0.2
        id_depth =  time_left()*0.3

        # Iterative deepening
        if game.move_count > 9:
            new_depth = 8
        else:
            new_depth = 4 #can go deep cos alpha-bet is in effect

        # best_move, utility_val = self.alphabeta(game, time_left, depth=new_depth)

        while (time_left() > id_depth and new_depth<50):  # keep looping as long as there is time
            new_depth = new_depth + 1
            best_move, utility_val = self.alphabeta(game, time_left, depth=new_depth)
            # quiescence_search.append(best_move)
            #quiesc_len = len(quiescence_search)
            # implement quiescence
            # if (len(quiescence_search)> 3):
            #      if (quiescence_search[-1] == quiescence_search[-2] == quiescence_search[-3]):
            #          break
                     #== quiescence_search[quiesc_len - 3]) and
        #     #             (quiescence_search[quiesc_len - 3] == quiescence_search[quiesc_len - 4]) and
        #     #             (quiescence_search[quiesc_len - 4] == quiescence_search[quiesc_len - 5])):
        #     #          break
        #
        # # premature execution
        # # if ((quiescence_search[quiesc_len - 1] != quiescence_search[quiesc_len - 2]) and (
        # #         quiescence_search[quiesc_len - 2]== quiescence_search[new_depth - 3] == quiescence_search[quiesc_len - 4])):
        # #         best_move = quiescence_search[quiesc_len - 2]  # if computation time is not enough to complete recursion
        # #         # and execution stops half was cos of time
        #
        # # change minimax to alphabeta after completing alphabeta part of assignment
        # #best_move, utility_val = self.minimax(game, time_left, depth=self.search_depth)
        # #best_move, utility_val = self.alphabeta(game, time_left, depth=self.search_depth)
        #
        # print best_move
        # print new_depth

        if best_move == (-1, -1) and len(legal_moves) > 0:
            return legal_moves[choice(range(len(legal_moves)))]

        return best_move

    def utility(self, game):
        """Can be updated if desired"""
        return self.eval_fn.score(game)
        
    def minimax(self, game, time_left, depth=3, maximizing_player=True): #breadth first
        """Implementation of the minimax algorithm
        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.
        Returns:
            (tuple, int): best_move, best_val
        """


        legal_moves = game.get_legal_moves() #get list of branches/moves
        if(len(legal_moves) <= 0):
            return ((-1, -1), 0) #check game state

        my_reward_list = []
        my_move_list = []

        # active_ply = game.__active_player__
        # ply1 = game.__player_1__

        #if(game.__active_player__ == game.__player_1__):
        for mov in legal_moves:
            game_forecast = game.forecast_move(mov)  # forecast to get a new board
            min_move, min_reward = self.minValue(game_forecast, depth-1,time_left)
            my_move_list.append(mov) #my_move_list.append(min_move)
            my_reward_list.append(min_reward)

        max_move = max(my_reward_list) #argmax
        max_index = my_reward_list.index(max_move)

        return (my_move_list[max_index], max_move)
        #
        # else: #min player
        #      for mov in legal_moves:
        #          game_forecast = game.forecast_move(mov)  # forecast to get a new board
        #          max_move, max_reward = self.maxValue(game_forecast, depth-1,time_left)
        #          my_move_list.append(mov) #my_move_list.append(min_move)
        #          my_reward_list.append(max_reward)
        #
        #      min_move = min(my_reward_list) #argmin
        #      min_index = my_reward_list.index(min_move)
        #      return my_move_list[min_index], min_move

        #TODO: finish this function!
        # raise NotImplementedError
        #return best_move, best_val

    # textbook algorithm method
    def minValue(self,game,depth,time_left):
        legal_moves = game.get_legal_moves()
        my_reward_list = []

        if(depth<=0 or time_left()<=self.backup_time): #terminal conditions
            for mov in legal_moves:
                game_forecast = game.forecast_move(mov)  # forecast to get a new board
                my_reward_list.append(self.utility(game_forecast))

            if(len(legal_moves)==0): #run out of moves, depriotize
                return ((-1, -1), -9223372036854775807) #-9223372036854775808

            min_reward = min(my_reward_list)
            min_index = my_reward_list.index(min_reward)
            return (legal_moves[min_index], min_reward) #return utility of current state

        # if (len(legal_moves) <= 0):  # if game gets to the end and there is still more depth value
        #     return (-1, -1), 9223372036854775807

        #not a terminal condition, keep looping
        move_reward = 9223372036854775807
        min_move = (-1, -1) # or (-1,-1)
        for mov in legal_moves:
            game_forecast = game.forecast_move(mov)  # forecast to get a new board
            max_move, max_reward = self.maxValue(game_forecast, depth-1,time_left)
            if(move_reward>max_reward): #move_reward = min(move_reward, max_reward)
                move_reward = max_reward
                min_move = mov

        return (min_move, move_reward)

    def maxValue(self, game, depth,time_left):
        legal_moves = game.get_legal_moves()
        my_reward_list = []

        if (depth <=0 or time_left()<=self.backup_time):  # terminal conditions
            for mov in legal_moves:
                game_forecast = game.forecast_move(mov)  # forecast to get a new board state
                my_reward_list.append(self.utility(game_forecast))

            if (len(legal_moves) == 0):  # run out of moves, depriotize
                return ((-1, -1), 9223372036854775806) #9223372036854775807

            max_reward = max(my_reward_list)
            max_index = my_reward_list.index(max_reward)
            return (legal_moves[max_index], max_reward)  # return utility of current state

        # if (len(legal_moves) <= 0):  # if game gets to the end and there is still more depth value
        #     return (-1, -1), -9223372036854775808

        # not a terminal condition, keep looping
        move_reward = -9223372036854775808
        max_move = (-1, -1)  # or (-1,-1)
        for mov in legal_moves:
            game_forecast = game.forecast_move(mov)  # forecast to get a new board
            min_move, min_reward = self.minValue(game_forecast, depth-1,time_left)
            if (min_reward > move_reward):  # move_reward = min(move_reward, max_reward)
                move_reward = min_reward
                max_move = mov

        return (max_move, move_reward)

    def alphabeta(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implementation of the alphabeta algorithm
        
        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        best_move, val = self.maxValue_alpha(game, depth, time_left, alpha, beta)

        # TODO: finish this function!
        #raise NotImplementedError
        return best_move, val

    # textbook algorithm method
    def minValue_alpha(self,game,depth,time_left, alpha, beta):
        legal_moves = game.get_legal_moves()
        my_reward_list = []

        if(depth<=0 or time_left()<=self.backup_time): #terminal conditions
            # for mov in legal_moves:
            #     game_forecast = game.forecast_move(mov)  # forecast to get a new board
            #     my_reward_list.append(self.utility(game_forecast))

            if (len(legal_moves) == 0):  # run out of moves, depriotize
                return ((-1, -1), -9223372036854775808)  # -9223372036854775808

            # if all(item == my_reward_list[0] for item in my_reward_list):
            #     if (game.__active_player__ == game.__player_2__):
            #         min_index = choice(range(len(my_reward_list)))
            #         min_reward = my_reward_list[min_index]
            #     else:
            #         min_index=0
            #         min_reward = my_reward_list[min_index]
            # else:
            # min_reward = min(my_reward_list)
            # min_index = my_reward_list.index(min_reward)
            #
            # return (legal_moves[min_index], min_reward) #return utility of current state

            return (self.utility(game), (0,0))

        # if (len(legal_moves) <= 0):  # if game gets to the end and there is still more depth value
        #     return (-1, -1), 9223372036854775807

        #not a terminal condition, keep looping
        move_reward = 9223372036854775807
        min_move = (-1, -1) # or (-1,-1)
        for mov in legal_moves:
            game_forecast = game.forecast_move(mov)  # forecast to get a new board
            max_move, max_reward = self.maxValue_alpha(game_forecast, depth-1,time_left, alpha, beta)
            if(max_reward<move_reward): #move_reward = min(move_reward, max_reward)
                move_reward = max_reward
                min_move = mov

            if(move_reward <= alpha):
                return (mov, move_reward)

            if(move_reward<beta):
                beta = move_reward

        return (min_move, move_reward)

    def maxValue_alpha(self, game, depth,time_left, alpha, beta):
        legal_moves = game.get_legal_moves()
        # my_reward_list = []

        if (depth <=0 or time_left()<=self.backup_time):  # terminal conditions
            # for mov in legal_moves:
            #     game_forecast = game.forecast_move(mov)  # forecast to get a new board state
            #     my_reward_list.append(self.utility(game_forecast))
            #
            # if (len(legal_moves) == 0):  # run out of moves, depriotize
            #     return ((-1, -1), 9223372036854775807)  # 9223372036854775807

            # if all(item == my_reward_list[0] for item in my_reward_list):
            #     if (game.__active_player__ == game.__player_2__):
            #         max_index = choice(range(len(my_reward_list)))
            #         max_reward = my_reward_list[max_index]
            #     else:
            #         max_index=0
            #         max_reward = my_reward_list[max_index]
            # else:
            # max_reward = max(my_reward_list)
            # max_index = my_reward_list.index(max_reward)
            #
            # return (legal_moves[max_index], max_reward)  # return utility of current state
            return (self.utility(game), (0,0))

        # if (len(legal_moves) <= 0):  # if game gets to the end and there is still more depth value
        #     return (-1, -1), -9223372036854775808

        # not a terminal condition, keep looping
        move_reward = -9223372036854775808
        max_move = (-1, -1)  # or (-1,-1)
        for mov in legal_moves:
            game_forecast = game.forecast_move(mov)  # forecast to get a new board
            min_move, min_reward = self.minValue_alpha(game_forecast, depth-1,time_left, alpha, beta)
            if (min_reward > move_reward):  # move_reward = min(move_reward, max_reward)
                move_reward = min_reward
                max_move =mov

            if(move_reward>=beta):
                return (mov, move_reward)

            if (alpha< move_reward):
                alpha = move_reward

        return (max_move, move_reward)