# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
from game import Actions
import random, util
import layout
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # print "legal action list: ", legalMoves
        # Choose one of the best actions
        # print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ get Action ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        scores = list()
        for action in legalMoves:
            aScore = self.evaluationFunction(gameState, action)
            scores.append(aScore)
            # print "^^^ aScore:", aScore
            # print "^^^ action:", action

        # scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        ghostPosition =  gameState.getGhostPositions()
        xGhost, yGhost =  ghostPosition[0]
        xPac, yPac = gameState.getPacmanPosition()

        if abs(xGhost-xPac)==1 and abs(yGhost-yPac)==1:
            return 'Stop'
        if abs(xGhost-xPac) == 0 and abs(yGhost-yPac) == 2:
            return 'Stop'
        if abs(xGhost-xPac) == 2 and abs(yGhost-yPac) == 0:
            return 'Stop' 

        # print "^^^ scores:", scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print "^^^ Pacman position:", gameState.getPacmanPosition()
        # print "^^^ Ghost position:", ghostPosition
        # print "all scores:", scores
        # print "all moves:", legalMoves
        # print "^^^ chosen move:", legalMoves[chosenIndex]
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # newPos = tuple()
        # print "new Ghost States: ", newGhostStates
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** distance to Ghost***"
        newGhostStates = successorGameState.getGhostStates()
        ghostPosition = newGhostStates[0].getPosition()
        distanceToNearestGhost = util.manhattanDistance(successorGameState.getPacmanPosition(), ghostPosition)
        
        "*** distance to food***"
        newPacmanPos = successorGameState.getPacmanPosition()
        # successorGameState.getWalls()
        newFood = successorGameState.getFood()
        actionList = self.breadthFirstSearch(currentGameState, newPacmanPos, newFood)
        distanceToNearestFood = len(actionList) - 1

        # print "+++++++++++++++++++ Evaluation Function ++++++++++++++++++++"
        # print "+++ Current position:", currentGameState.getPacmanPosition()
        # print "+++ Successor's position:", newPacmanPos
        # print "actions to closestFood:", actionList
        # print "successorGameState.getScore", successorGameState.getScore()
        # print "distanceToNearestFood:", distanceToNearestFood
        # print "distanceToNearestGhost:", distanceToNearestGhost
        if distanceToNearestGhost <= 1:
            return -9999

        score  = successorGameState.getScore() - .25*distanceToNearestFood + 1/distanceToNearestGhost

        # print "score:", score
        return score

    def breadthFirstSearch(self, currentGameState, startPos, foodGrid):
        # print "--------------- breadthFirstSearch ---------------"
        fringe = util.Queue()
        fringe.push( (startPos, []) )
        visited = []
        # visitedSet = set()
        while not fringe.isEmpty():
            currentCoor, listActions = fringe.pop()
            # print "****current Coor: ", currentCoor
            for coor, action in self.getSuccessors(currentGameState, currentCoor):
                x,y = coor
                # print "coor: ", coor
                # print "x:",x
                # print "y:",y
                if foodGrid[x][y]:
                    return listActions + [action]
                elif coor not in visited:
                    fringe.push( (coor, listActions + [action]))
                    visited.append(coor)
                    # print listActions + [action]
                else:
                    pass
        return []

    def getSuccessors(self, currentGameState, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        # print "************** get Succesors *******************"
        walls = currentGameState.getWalls()
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # print "**** state: ", state
            # print "**** legal action: ", action
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not walls[nextx][nexty]:
                nextState = (nextx, nexty)
                # print "+++ nextState: ", nextState
                successors.append( ( nextState, action) )
                # currentGameState.ge
        return successors

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

