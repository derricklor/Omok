# Omok game program with PLAYER VS AI

# Import and initialize the pygame library
import pygame
import math
import numpy as np
import copy as cp


# Define available keys
from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 100, 100)
PIECE_SIZE = 100
EMPTY = 0
PLAYER = 1
ENEMY = 2
DRAW = 3

class Board():
    def __init__(self, size, winLength):
        if winLength > size:
            print("Win length cannot be greater than board!")
            exit(0)
        elif winLength < 4:
            #win length < 4 then there is guaranteed win for piece that moves first
            print("Win length cannot be less than 4!")
            exit(0)
        self.boardSize = size
        self.winLength = winLength
        self.board = np.zeros((size,size))
        self.availableActions = []  #Amortized analysis, do more work early to get fast results later
        for row in range(size):
            for col in range(size):
                self.availableActions.append((row, col))
        self.turn = PLAYER
        self.piecesOnBoard = 0
        
    # Reset the game board back to beginning
    def reset(self):
        self.board = np.zeros((self.boardSize,self.boardSize))
        self.availableActions = []
        for row in range(self.boardSize):
            for col in range(self.boardSize):
                self.availableActions.append((row, col))
        self.turn = PLAYER
        self.piecesOnBoard = 0

        

    # Bool to check if move is valid
    def isValidMove(self, row, col):
        if (row,col) in self.availableActions:
            return True
        else:
            return False
    
    # Place the piece of whoever's turn it is
    def placePiece(self, row, col):
        self.board[row][col] = self.turn
        self.piecesOnBoard += 1
        self.availableActions.remove((row, col))
        
    def switchTurns(self):
        #flip flop between PLAYER and ENEMY
        if self.turn == PLAYER:
            self.turn = ENEMY
        else:
            self.turn = PLAYER

    # Used solely in evalPosition.
    # Evaluate the window based on number of twos, threes and
    # fours in a row depending on whose turn it is.
    def evalWindow(self, window):
        score = 0
        if self.turn == PLAYER:
            piece = PLAYER
            opponentPiece = ENEMY
        else:
            piece = ENEMY
            opponentPiece = PLAYER
        #if window.count(piece) == self.winLength:
        #    score += 1000
        if window.count(piece) == self.winLength-1 and window.count(EMPTY) == 1:
            score += 100    #best score in my testing
        elif window.count(piece) == self.winLength-2 and window.count(EMPTY) == 2:
            score += 10     #best score in my testing
        elif window.count(piece) == self.winLength-3 and window.count(EMPTY) == 3:
            score += 5      #best score in my testing

        if window.count(opponentPiece) == self.winLength-1 and window.count(EMPTY) == 1:
            score -= 200    #best score in my testing
        elif window.count(opponentPiece) == self.winLength-2 and window.count(EMPTY) == 2:
            score -= 50     #best score in my testing

        return score

    # Evaluate the board position of current turn by creating winLength size windows
    # then calling evalWindow to score the window, for whole board
    def evalPosition(self):
        #copy same structure from isWon
        score = 0

        #horizontal
        for row in range(0, self.boardSize):
            #get the entire row then,
            #slice into list of len 5 [0:5],[1:6],[2:7] ... etc
            for col in range(0, self.winLength):
                a = self.board[row, 0+col:self.winLength+col]
                b = list(a)
                score += self.evalWindow(b)
                
        #vertical
        #rotate 90 degrees and rerun code from horizontal to check vertical
        tempBoard = np.rot90(self.board)
        for row in range(0, self.boardSize):
        #get the entire column then slice into list of winLength
            for col in range(0, self.winLength):
                a = tempBoard[row, 0+col:self.winLength+col]
                b = list(a)
                score += self.evalWindow(b)
        
        #check all pos slope diagonals, since flipped on xaxis it looks like backwards slash \\\ right side up
        for row in range(0, self.boardSize-self.winLength+1): #+1 because inrange excludes on end
            for col in range(0, self.boardSize-self.winLength+1): #+1 because inrange excludes on end
                b = list()
                for i in range(0, self.winLength):
                    b.append(self.board[row+i][col+i])
                score += self.evalWindow(b)


        #check all neg slope diagonals, looks like forward slash /// right side up
        for row in range(self.winLength-1, self.boardSize): #-1 because 0 index
            for col in range(0, self.boardSize-self.winLength+1): #+1 because inrange excludes on end
                b = list()
                for i in range(0, self.winLength):
                    b.append(self.board[row-i][col+i])
                score += self.evalWindow(b)

        return score
                    

    # Ai algorithm, returns row,col to move
    def minimax(self, depth, maximizingPlayer):
        #check all the possible terminal states: hit search depth, draw, or won
        #and return the heuristic eval
        won, winner = self.isWon()
        if won:
            if self.turn == winner:
                return None, None, 10_000
            else:
                return None, None, -10_000
        elif self.isDraw(): # drawed game
            return None, None, -100
        elif depth == 0:
            return None, None, self.evalPosition() # score current position of current turn
            
        if maximizingPlayer:
            value = -np.infty
            index = np.random.randint(len(self.availableActions)) 
            bestrow, bestcol = self.availableActions[index]
            for action in self.availableActions:
                #dont need to check if isValidMove
                boardCopy = cp.deepcopy(self)
                boardCopy.placePiece(action[0], action[1])
                #boardCopy.switchTurns() 
                _, _, newScore = boardCopy.minimax(depth-1, False)
                if newScore > value:
                    value = newScore
                    bestrow = action[0]
                    bestcol = action[1]
            return bestrow, bestcol, value

        else:   #minimizing player
            value = np.infty
            index = np.random.randint(len(self.availableActions)) 
            bestrow, bestcol = self.availableActions[index]
            for action in self.availableActions:
                #dont need to check if isValidMove
                boardCopy = cp.deepcopy(self)
                boardCopy.placePiece(action[0], action[1])
                #boardCopy.switchTurns() 
                _, _, newScore = boardCopy.minimax(depth-1, True)
                if newScore < value:
                    value = newScore
                    bestrow = action[0]
                    bestcol = action[1]
            return bestrow, bestcol, value


    # Ai algorithm using minimax with alpha beta pruning, returns row,col to move
    def minimaxABPruning(self, depth, alpha, beta, maximizingPlayer):
        #check all the possible terminal states: hit search depth, draw, or won
        #and return the heuristic eval
        
        won, winner = self.isWon()
        if won:
            if self.turn == winner:
                return None, None, 10_000
            else:
                return None, None, -10_000
        elif self.isDraw(): # drawed game
            return None, None, -100
        elif depth == 0:
            return None, None, self.evalPosition() # score current position of current turn
            
        if maximizingPlayer:
            value = -np.infty
            index = np.random.randint(len(self.availableActions)) 
            bestrow, bestcol = self.availableActions[index]
            for action in self.availableActions:
                #dont need to check if isValidMove
                boardCopy = cp.deepcopy(self)
                boardCopy.placePiece(action[0], action[1])
                #boardCopy.switchTurns() 
                _, _, newScore = boardCopy.minimaxABPruning(depth-1, alpha, beta, False)
                if newScore > value:
                    value = newScore
                    bestrow = action[0]
                    bestcol = action[1]

                alpha = max(alpha, value)
                if alpha >= beta:
                    break
                    
            return bestrow, bestcol, value

        else:   #minimizing player
            value = np.infty
            index = np.random.randint(len(self.availableActions)) 
            bestrow, bestcol = self.availableActions[index]
            for action in self.availableActions:
                #dont need to check if isValidMove
                boardCopy = cp.deepcopy(self)
                boardCopy.placePiece(action[0], action[1])
                #boardCopy.switchTurns() 
                _, _, newScore = boardCopy.minimaxABPruning(depth-1, alpha, beta, True)
                if newScore < value:
                    value = newScore
                    bestrow = action[0]
                    bestcol = action[1]

                beta = min(beta, value)
                if alpha >= beta:
                    break
                    
            return bestrow, bestcol, value
            

    # Strictly check board for a win. Does not check for draw.
    # ALWAYS CHECK isWon BEFORE isDraw.
    # returns bool, int
    def isWon(self):

        #check all horizontals
        for row in range(0, self.boardSize):
            #get the entire row then
            #slice into list of len 5 [0:5],[1:6],[2:7] ... etc
            for col in range(0, self.winLength):
                a = self.board[row, 0+col:self.winLength+col]
                b = list(a)
                if b.count(PLAYER) >= self.winLength:
                    return True, PLAYER
                if b.count(ENEMY) >= self.winLength:
                    return True, ENEMY
                
        #check all verticals
        #rotate 90 degrees and rerun code from horizontal to check vertical
        tempBoard = np.rot90(self.board)
        for row in range(0, self.boardSize):
        #get the entire column then slice into list of winLength=5
            for col in range(0, self.winLength):
                a = tempBoard[row, 0+col:self.winLength+col]
                b = list(a)
                if b.count(PLAYER) >= self.winLength:
                    return True, PLAYER
                if b.count(ENEMY) >= self.winLength:
                    return True, ENEMY
        
        
        #check all pos slope diagonals, since flipped on xaxis it looks like backwards slash \\\ right side up
        for row in range(0, self.boardSize-self.winLength+1): #+1 because inrange excludes on end
            for col in range(0, self.boardSize-self.winLength+1): #+1 because inrange excludes on end
                a = list()
                for i in range(0, self.winLength):
                    a.append(self.board[row+i][col+i])
                if a.count(PLAYER) >= self.winLength:
                    return True, PLAYER
                if a.count(ENEMY) >= self.winLength:
                    return True, ENEMY

        #check all neg slope diagonals, looks like forward slash /// right side up
        for row in range(self.winLength-1, self.boardSize): #-1 because 0 index
            for col in range(0, self.boardSize-self.winLength+1): #+1 because inrange excludes on end
                a = list()
                for i in range(0, self.winLength):
                    a.append(self.board[row-i][col+i])
                if a.count(PLAYER) >= self.winLength:
                    return True, PLAYER
                if a.count(ENEMY) >= self.winLength:
                    return True, ENEMY
                
        
        #default case, no winner and len(availableActions) > 0
        return False, 0

    # Checks for draw. Returns bool.
    # ALWAYS CHECK isWon BEFORE isDraw.
    def isDraw(self):
        if len(self.availableActions) <= 0:
            return True
        else:
            return False


    def printBoard(self):
        print(self.board)

    #This function picks a random square within a given distance from the center square.
    #center_x,  center_y: tuple containing the coordinates of center
    #max_distance: The maximum distance from the center square (defaults to 1).
    # Returns:A tuple containing the coordinates of a random neighbor square.
    def get_random_neighbor(self, center_x, center_y, max_distance):
        if len(self.availableActions) == 0:
            return (None, None)
        while True:
            dx = np.random.randint(-max_distance, max_distance+1)
            dy = np.random.randint(-max_distance, max_distance+1)
            new_x = center_x + dx # col
            new_y = center_y + dy # row

            # Check if the new coordinate is in the availableActions
            coord = (new_y, new_x)
            if coord in self.availableActions:
                return coord
                
    # Similar to Gymnasium environment where we take action tuple (row, col) and
    # play the move for the agent in this blackbox, where state, reward, running, and winner
    def step(self, action):

        #player move
        row, col = action
        self.placePiece(row, col)
        
        won, winner = self.isWon()
        if won:
            if winner == PLAYER:
                return self.board, 100, False, PLAYER
            else:
                return self.board, -100, False, ENEMY
        elif self.isDraw():
            return self.board, -10, False, DRAW
        else:
            self.switchTurns()

        #ai move
        if self.piecesOnBoard < 2:
            coord = self.get_random_neighbor(4, 4, 1)
            rowAI, colAI = coord
        else:
            rowAI, colAI, score = self.minimaxABPruning(1, -np.infty, np.infty, True)
        self.placePiece(rowAI, colAI)
        
        won, winner = self.isWon()
        if won:
            if winner == PLAYER:
                return self.board, 100, False, PLAYER
            else:
                return self.board, -100, False, ENEMY
        elif self.isDraw():
            return self.board, -10, False, DRAW
        else:
            self.switchTurns()

        #default
        return self.board, -1, True, None
    
        

# Define the Player object by extending pygame.sprite.Sprite
class Player(pygame.sprite.Sprite):
    def __init__(self,x ,y):
        super(Player, self).__init__()
        self.surf = pygame.image.load("black.png").convert_alpha()
        #self.surf.set_colorkey(WHITE, RLEACCEL) # Used to set color transparent
        self.rect = pygame.Rect(x,y,0,0)

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(Enemy, self).__init__()
        self.surf = pygame.image.load("white.png").convert_alpha()
        #self.surf.set_colorkey(WHITE, RLEACCEL) # Used to set color transparent
        self.rect = pygame.Rect(x,y,0,0)

class Background(pygame.sprite.Sprite):
    def __init__(self):
        super(Background, self).__init__()
        self.surf = pygame.image.load("board.png").convert()
        #self.surf.set_colorkey((0, 0, 0), RLEACCEL) # Used to set color transparent
        self.rect = self.surf.get_rect() #default (0,0) top left corner

def PlayerVSPlayer(board):
    pygame.init()
    # Set up the drawing window
    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 900
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    background = Background()
    screen.blit(background.surf, background.rect)
    myfont = pygame.font.SysFont("monospace", 75)
    # Run until the user asks to quit
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x = int(math.floor(x/PIECE_SIZE))
                y = int(math.floor(y/PIECE_SIZE))
                if board.isValidMove(y, x): # y = row, x = col
                    board.placePiece(y, x)
                    if board.turn == PLAYER:
                        piece = Player(x*PIECE_SIZE, y*PIECE_SIZE)
                    else:
                        piece = Enemy(x*PIECE_SIZE, y*PIECE_SIZE)
                    screen.blit(piece.surf, piece.rect)
                    won, winner = board.isWon()#check for win or draw
                    if won:
                        string = "Player "+ str(winner) +" Won!"
                        label = myfont.render(string, 1, RED)
                        screen.blit(label,(100,100))
                        pygame.display.flip()
                        pygame.time.wait(3000)
                        board.reset()
                        screen.blit(background.surf, background.rect)
                        running = False
                    elif board.isDraw():
                        string = "Draw!"
                        label = myfont.render(string, 1, RED)
                        screen.blit(label,(100,100))
                        pygame.display.flip()
                        pygame.time.wait(3000)
                        board.reset()
                        screen.blit(background.surf, background.rect)
                        running = False
                    else:
                        #else keep playing
                        board.switchTurns()
                    
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
        
            # Flip the display
            pygame.display.flip()
    # Done! Time to quit.
    pygame.quit()
    
def PlayerVSEnemy(board):
    pygame.init()
    # Set up the drawing window
    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 900
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    background = Background()
    screen.blit(background.surf, background.rect)
    myfont = pygame.font.SysFont("monospace", 75)
    # Run until the user asks to quit
    running = True
    while running:
        if board.turn == PLAYER:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    x = int(math.floor(x/PIECE_SIZE))
                    y = int(math.floor(y/PIECE_SIZE))
                    if board.isValidMove(y, x): # y = row, x = col
                        board.placePiece(y, x)
                        piece = Player(x*PIECE_SIZE, y*PIECE_SIZE)
                        screen.blit(piece.surf, piece.rect)
                        won, winner = board.isWon()#check for win or draw
                        if won:
                            string = "Player "+ str(winner) +" Won!"
                            label = myfont.render(string, 1, RED)
                            screen.blit(label,(100,100))
                            pygame.display.flip()
                            pygame.time.wait(3000)
                            board.reset()
                            screen.blit(background.surf, background.rect)
                            running = False
                        elif board.isDraw():
                            string = "Draw!"
                            label = myfont.render(string, 1, RED)
                            screen.blit(label,(100,100))
                            pygame.display.flip()
                            pygame.time.wait(3000)
                            board.reset()
                            screen.blit(background.surf, background.rect)
                            running = False
                        else:
                            #else keep playing
                            board.switchTurns()
                        
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
        else:#enemy ai turn
            pygame.time.wait(1000)

            if board.piecesOnBoard < 2:
                coord = board.get_random_neighbor(4, 4, 1)
                ro, co = coord
                board.placePiece(ro, co)
            else:
                ro, co, score = board.minimaxABPruning(1, -np.infty, np.infty,True)
                #print(f'{ro+1}, {co+1} is the best move for ENEMY')
                board.placePiece(ro, co)

            piece = Enemy(co*PIECE_SIZE, ro*PIECE_SIZE)
            screen.blit(piece.surf, piece.rect)
            #check for win or draw
            won, winner = board.isWon()
            if won:
                string = "Player "+ str(winner) +" Won!"
                label = myfont.render(string, 1, RED)
                screen.blit(label,(100,100))
                pygame.display.flip()
                pygame.time.wait(3000)
                board.reset()
                screen.blit(background.surf, background.rect)
                running = False
            elif board.isDraw():
                string = "Draw!"
                label = myfont.render(string, 1, RED)
                screen.blit(label,(100,100))
                pygame.display.flip()
                pygame.time.wait(3000)
                board.reset()
                screen.blit(background.surf, background.rect)
                running = False
                #else keep playing
            else:
                board.switchTurns()
        # Flip the display
        pygame.display.flip()
    # Done! Time to quit.
    pygame.quit()
    
def EnemyVSEnemy(board):
    pygame.init()
    # Set up the drawing window
    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 900
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    background = Background()
    screen.blit(background.surf, background.rect)
    myfont = pygame.font.SysFont("monospace", 75)
    # Run until the user asks to quit
    running = True
    while running:
        if board.turn == PLAYER:
            pygame.time.wait(500)

            if board.piecesOnBoard < 1:
                coord = board.get_random_neighbor(4, 4, 3)
                ro, co = coord
                board.placePiece(ro, co)
            else:
                #ro, co, score = board.minimax(1, True)
                ro, co, score = board.minimaxABPruning(1, -np.infty, np.infty, True)
                #print(f'P 1 best move: {ro+1}, {co+1} with score: {score}')
                board.placePiece(ro, co)

            piece = Player(co*PIECE_SIZE, ro*PIECE_SIZE)
            screen.blit(piece.surf, piece.rect)
            #check for win or draw
            
            won, winner = board.isWon()
            if won:
                string = "Player "+ str(winner) +" Won!"
                label = myfont.render(string, 1, RED)
                screen.blit(label,(100,100))
                pygame.display.flip()
                pygame.time.wait(3000)
                board.reset()
                screen.blit(background.surf, background.rect)
                running = False
            elif board.isDraw():
                string = "Draw!"
                label = myfont.render(string, 1, RED)
                screen.blit(label,(100,100))
                pygame.display.flip()
                pygame.time.wait(3000)
                board.reset()
                screen.blit(background.surf, background.rect)
                running = False
                #else keep playing
            else:
                board.switchTurns()
        else:#enemy ai turn
            pygame.time.wait(500)

            if board.piecesOnBoard < 2:
                coord = board.get_random_neighbor(4, 4, 3)
                ro, co = coord
                board.placePiece(ro, co)
            else:
                #ro, co, score = board.minimax(1, False)
                ro, co, score = board.minimaxABPruning(1, -np.infty, np.infty,True)
                #print(f'P 2 best move: {ro+1}, {co+1} with score: {score}')
                board.placePiece(ro, co)

            piece = Enemy(co*PIECE_SIZE, ro*PIECE_SIZE)
            screen.blit(piece.surf, piece.rect)
            #check for win or draw
            
            won, winner = board.isWon()
            if won:
                string = "Player "+ str(winner) +" Won!"
                label = myfont.render(string, 1, RED)
                screen.blit(label,(100,100))
                pygame.display.flip()
                pygame.time.wait(3000)
                board.reset()
                screen.blit(background.surf, background.rect)
                running = False
            elif board.isDraw():
                string = "Draw!"
                label = myfont.render(string, 1, RED)
                screen.blit(label,(100,100))
                pygame.display.flip()
                pygame.time.wait(3000)
                board.reset()
                screen.blit(background.surf, background.rect)
                running = False
                #else keep playing
            else:
                board.switchTurns()
        # Flip the display
        pygame.display.flip()
    # Done! Time to quit.
    pygame.quit()

def PlayOmokPygame(arg1, arg2):
    # Set up board
    board = Board(9, 5)
    
    if arg1 == PLAYER and arg2 == PLAYER:
        PlayerVSPlayer(board)
    elif arg1 == PLAYER and arg2 == ENEMY:
        PlayerVSEnemy(board)
    elif arg1 == ENEMY and arg2 == PLAYER:
        board.switchTurns()
        PlayerVSEnemy(board)
    elif arg1 == ENEMY and arg2 == ENEMY:
        EnemyVSEnemy(board)
    


    