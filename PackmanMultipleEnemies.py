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


from game import Directions
import random, util

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

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # print "BestScore: ", bestScore, " BestIndices: ", bestIndices
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        print "Legal Moves: ", legalMoves
        print "Scores-: ", scores, " Best: ", bestScore, " chosen: ",legalMoves[chosenIndex], " I: ",chosenIndex
        print "---------"
        "Add more of your code here if you want to"

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
        newPos = successorGameState.getPacmanPosition()
        newPosX, newPosY = newPos
        newGhostStates = successorGameState.getGhostStates()
        score = 0.0
        currentPosition = currentGameState.getPacmanPosition()
        currentPosX, currentPosY = currentPosition
        currentStateFoodList = currentGameState.getFood().asList()
        MAX_POINTS = 1000000000
        FAVOUR_POINTS = 50

        # Winning game state gets maximum score
        if successorGameState.isWin():
                return MAX_POINTS

        # Next food count is less than current
        if currentGameState.getNumFood() > successorGameState.getNumFood():
                score += FAVOUR_POINTS

        # If in next state, a capsule is eaten, then it is favourable
        if len(currentGameState.getCapsules()) > len(successorGameState.getCapsules()):
                score += FAVOUR_POINTS

        # Add the distance of ghost from current pacman position in the score
        for thisGhostState in newGhostStates:
            ghostPosition = thisGhostState.getPosition()
            ghostPosX, ghostPosY = ghostPosition
            ghostDistance = abs(ghostPosX - currentPosX) + abs(ghostPosY - currentPosY)
            if ghostPosition == newPos:
                return -MAX_POINTS

            score += ghostDistance

        # Find the closest food
        foodDistance = MAX_POINTS
        for foodCoordinate in currentStateFoodList:
            foodX, foodY = foodCoordinate
            thisDistance = abs(foodX - newPosX) + abs(foodY - newPosY)
            foodDistance = min(foodDistance,thisDistance)
            if Directions.STOP in action:
                return -MAX_POINTS
        # Minimum food distance should be given a higher score, therefore finding its inverse
        score += 1.0/(1.0 + foodDistance)

        return score

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
        self.BEST_POINTS = 100000000
        self.WORST_POINTS = -100000000
        self.noOfAgents = gameState.getNumAgents()
        return self.maximumValue(0, gameState, 0)[0]

    # Iterating over depth, until we find leaf nodes. On reaching depth and finding the leaf node, we calculate the score by
    # calling the evaluation function for the state and returning it. For, all other states we find max or min for them, depending
    # on whether it is a pacman or ghost. For each depth, we will consider all agents, i.e., pacman and the number of ghosts
    # and then move to the next level.
    def minimax(self, indexOfAgent, state, depth):
        maxDepth = self.depth
        # If a leaf is reached or is a win state or is a lose state
        if depth is  maxDepth * self.noOfAgents  or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        if indexOfAgent != 0:
            # For ghosts, min value must be found
            return self.minimumValue(indexOfAgent, state,  depth)[1]
        else:
            # Since pacman is a max agent, max value must be found for pacman
            return self.maximumValue(indexOfAgent, state, depth)[1]


    def minimumValue(self, indexOfAgent, state, depth):
        # Initializing optimum action with the max value possible
        optimumAction = ("minimum", self.BEST_POINTS)
        newDepth = depth + 1
        legalAgentActions = state.getLegalActions(indexOfAgent)

        # Iterating over all legal actions of an agent, finding its successor
        # and checking the minimum possible value among all successors (since they are ghosts, hence minimum)
        for action in legalAgentActions:
            newAgentIndex = newDepth % self.noOfAgents
            # Generates successor based on agentIndex and action of agent
            successor = state.generateSuccessor(indexOfAgent, action)
            # For this action, get the min value from the successors
            actionOfSuccessor = (action,self.minimax(newAgentIndex, successor, newDepth))
            # Optimum action will be the one with the min value from the successors
            optimumAction = min(optimumAction, actionOfSuccessor, key = lambda item:item[1])

        return optimumAction

    def maximumValue(self, indexOfAgent, state, depth):
        # Initializing optimum action with the lowest value possible
        optimumAction = ("maximum", self.WORST_POINTS)
        newDepth = depth + 1
        legalAgentActions = state.getLegalActions(indexOfAgent)

        # Iterating over all legal actions of an agent, finding its successor
        # and checking the maximum possible value among all successors (since it is pacman, hence maximum)
        for action in legalAgentActions:
            newAgentIndex = newDepth % self.noOfAgents
            # Generates successor based on agentIndex and action of agent
            successor = state.generateSuccessor(indexOfAgent, action)
            # For this action, get the max value from the successors
            actionOfSuccessor = (action, self.minimax(newAgentIndex, successor, newDepth))
            # Optimum action will be the one with the max value from the successors
            optimumAction = max(optimumAction, actionOfSuccessor, key=lambda item:item[1])

        return optimumAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Alpha is given the lowest value
        self.WORST_POINTS = -100000000
        # Beta is given the max value
        self.BEST_POINTS = 100000000
        self.noOfAgents = gameState.getNumAgents()

        # We need the best minimax value for pacman being the max agent
        action = self.maximumValue(0, gameState, 0, self.WORST_POINTS, self.BEST_POINTS)
        return action[0]

    def alphabetapruning(self, indexOfAgent, state , depth, alpha, beta):
        maxDepth = self.depth
        noOfAgents = state.getNumAgents()

        # If a leaf is reached or is a win state or is a lose state
        if depth is maxDepth * noOfAgents or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        if indexOfAgent != 0:
            # For ghosts being min agents at index > 0, return the minValue that can be attained from the successors
            value = self.minimumValue(indexOfAgent, state, depth, alpha, beta)
            return value[1]
        else:
            # For pacman being the max agent at index 0, return the maxValue that can be attained from the successors
            value = self.maximumValue(indexOfAgent, state, depth, alpha, beta)
            return value[1]

    def minimumValue(self, indexOfAgent, state, depth, alpha, beta):
        # Initializing optimum action with the max value possible
        optimumAction = ("minimum", self.BEST_POINTS)
        newDepth = depth + 1
        legalAgentActions = state.getLegalActions(indexOfAgent)

        for action in legalAgentActions:
            newIndex = newDepth % self.noOfAgents
            # Generates successor based on agentIndex and action of agent
            successor = state.generateSuccessor(indexOfAgent, action)
            # For this action, get the min value from the successors
            actionOfSuccessor = (action, self.alphabetapruning(newIndex, successor ,newDepth, alpha, beta))
            # Optimum action will be the one with the min value from the successors
            optimumAction = min(optimumAction, actionOfSuccessor, key = lambda item:item[1])

            # Pruning the sub tree is greater than alpha, we take min of beta and optimum action,
            # otherwise return the optimum action
            if optimumAction[1] >= alpha:
                beta = min(beta, optimumAction[1])
            else:
                return optimumAction

        return optimumAction

    def maximumValue(self, indexOfAgent, state, depth, alpha, beta):
        # Initializing optimum action with the lowest value possible
        optimumAction = ("maximum", self.WORST_POINTS)
        newDepth = depth + 1
        legalAgentActions = state.getLegalActions(indexOfAgent)

        for action in legalAgentActions:
            newIndex = newDepth % self.noOfAgents
            # Generates successor based on agentIndex and action of agent
            successor = state.generateSuccessor(indexOfAgent, action)
            # For this action, get the max value from the successors
            actionOfSuccessor = (action, self.alphabetapruning(newIndex, successor, newDepth, alpha, beta))
            # Optimum action will be the one with the max value from the successors
            optimumAction = max(optimumAction, actionOfSuccessor, key=lambda item:item[1])

            # Pruning the sub tree is less than beta, we take max of alpha and optimum action,
            # otherwise return the optimum action
            if optimumAction[1] <= beta:
                alpha = max(alpha , optimumAction[1])
            else:
                return optimumAction

        return optimumAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
        # Giving the max depth for expectimax
        self.WORST_POINTS = -100000000
        self.BEST_POINTS = 100000000
        self.noOfAgents = gameState.getNumAgents()
        maximumDepth = self.depth * self.noOfAgents
        action = self.expectimax("expected", gameState,  maximumDepth, 0)
        return action[0]

    def expectimax(self, action, state, depth, indexOfAgent):

        # If the depth is 0 or the game state is win or lose
        if depth is 0 or state.isLose() or state.isWin():
            return (action, self.evaluationFunction(state))

        if indexOfAgent != 0:
            # For ghost being the chance node, return the expected value
            return self.expectedValue(action, state, depth, indexOfAgent)
        else:
            # For pacman being the 0th agent, return the max value
            return self.maximumValue(action, state, depth, indexOfAgent)

    def maximumValue(self, action, state, depth, indexOfAgent):

        # Initialize the optimum action with the lowest value
        optimumAction = ("maximum", self.WORST_POINTS)
        newDepth = depth - 1
        nextAgentIndex = indexOfAgent + 1
        nextAgent = nextAgentIndex % self.noOfAgents
        legalAgentActions = state.getLegalActions(indexOfAgent)

        for allowedAction in legalAgentActions:
            actionOfSuccessor = None

            if depth != self.depth * self.noOfAgents:
                actionOfSuccessor = action
            else:
                actionOfSuccessor = allowedAction
            # Generates successor based on agentIndex and action of agent
            successor = state.generateSuccessor(indexOfAgent, allowedAction)
            actionOfSuccessor = self.expectimax(actionOfSuccessor, successor, newDepth, nextAgent)
            # Optimum action will be the one with the  max value from the successors
            optimumAction = max(actionOfSuccessor, optimumAction, key = lambda item:item[1])

        return optimumAction

    def expectedValue(self, action, state, depth, indexOfAgent):
        # Initial average score is zero
        avgScore = 0
        nextAgentIndex = (indexOfAgent + 1)
        allowedActions = state.getLegalActions(indexOfAgent)
        lengthAllowedActions = len(allowedActions)
        # Taking probability as under root of length of allowed actions
        prob = pow(lengthAllowedActions, 1/2)
        actionTuple = ()
        actionTuple += (action,)
        newDepth = depth - 1
        for allowedAction in allowedActions:
            nextAgent =  nextAgentIndex % self.noOfAgents
            # Generates successor based on agentIndex and action of agent
            successor = state.generateSuccessor(indexOfAgent, allowedAction)
            # Getting the average value for chance node successors
            optimumAction = self.expectimax(action, successor, newDepth , nextAgent)
            avgScore += prob * optimumAction[1]

        actionTuple += (avgScore,)
        return actionTuple

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
      Evaluate state by  :
            * closest food
            * food left
            * capsules left
            * distance to ghost
    """
    "*** YOUR CODE HERE ***"

# Abbreviation
better = betterEvaluationFunction

