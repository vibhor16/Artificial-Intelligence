# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    ## queue --> To push the nodes in the queue along with the direction, visited --> List for storing visited nodes,
    ## Initial starting node...Push this node in the queue
    startNode = problem.getStartState()
    stack = util.Stack()
    visited = []
    directionVisited = {}
    stack.push(startNode)
    directionVisited[startNode] = []
    pathTillPoppedNode = []
    ## 1. Pop each element out of the queue until queue is empty
    ## 2. If the popped node is goal state, exit the loop and return the path
    ## 3. If not, check if popped node is in visited nodes. If it is not, add it to visited nodes and proceed to step 4.
    ## 4. Get the successors of popped node.
    ## 5. For each successor, if the successor is not in visited nodes, get the path covered till the successor node.
    ## 6. Push the successor node in the stack and the path in the direction visited list.
    ## 7. Repeat steps 1-7 until stack is empty.
    while not stack.isEmpty():
        poppedNode = stack.pop()
        pathTillPoppedNode = directionVisited[poppedNode]
        if problem.isGoalState(poppedNode):
            break;
        if poppedNode not in visited:
            visited.append(poppedNode)
            successors = problem.getSuccessors(poppedNode)
            for successorNode in successors:
                successorNodeElement = successorNode[0]
                successorNodePath = successorNode[1]
                pathCoveredBySuccNode = pathTillPoppedNode + [successorNodePath]
                stack.push(successorNodeElement)
                directionVisited[successorNodeElement] = pathCoveredBySuccNode

    return pathTillPoppedNode
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    ## stack --> To push the nodes in the stack, visited --> List for storing visited nodes,
    ## directionVisited --> Store the direction of visited nodes which is updated on every successor node
    ## Initial starting node...Push this node in stack
    startNode = problem.getStartState()
    queue = util.Queue()
    visited = []
    directions = []
    queue.push((startNode, directions))
    pathTillPoppedNode = []
    ## 1. Pop each element out of the stack until stack is empty
    ## 2. If the popped node is goal state, exit the loop and return the path
    ## 3. If not, check if popped node is in visited nodes. If it is not, add it to visited nodes and proceed to step 4.
    ## 4. Get the successors of popped node.
    ## 5. For each successor get the path covered till the successor node.
    ## 6. Push the successor node in the queue along with the path.
    ## 7. Repeat steps 1-7 until queue is empty.
    while not queue.isEmpty():
        poppedNodeData = queue.pop()
        poppedNode = poppedNodeData[0]
        pathTillPoppedNode = poppedNodeData[1]
        if problem.isGoalState(poppedNode):
            break
        if poppedNode not in visited:
            visited.append(poppedNode)
            successors = problem.getSuccessors(poppedNode)
            for successorNode in successors:
                successorNodeElement = successorNode[0]
                if successorNodeElement not in visited:
                    # Map parent for this successor
                    successorNodePath = successorNode[1]
                    pathCoveredBySuccNode = pathTillPoppedNode + [successorNodePath]
                    queue.push((successorNodeElement, pathCoveredBySuccNode))

    return pathTillPoppedNode
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    ## priority queue --> To push the nodes in the stack according to edge priority,
    ## visited --> List for storing visited nodes,
    ## Initial starting node...Push this node in priority queue along with empty directions and edge weight 0
    startNode = problem.getStartState()
    queue = util.PriorityQueue()
    visited = []
    directions = []
    queue.push((startNode, directions, 0), 0)
    pathTillPoppedNode = []
    ## 1. Pop each element out of the priority queue until it is empty
    ## 2. If the popped node is goal state, exit the loop and return the path
    ## 3. If not, check if popped node is in visited nodes. If it is not, add it to visited nodes and proceed to step 4.
    ## 4. Get the successors of popped node.
    ## 5. For each successor get the path covered and cumulative weight till the successor node.
    ## 6. Push the successor node in the priority queue along with the path and cumulative weight.
    ## 7. Repeat steps 1-7 until the priority queue is empty.
    while not queue.isEmpty():
        poppedNodeData = queue.pop()
        poppedNode = poppedNodeData[0]
        pathTillPoppedNode = poppedNodeData[1]
        cumulativeWeightTillPoppedNode = poppedNodeData[2]
        if problem.isGoalState(poppedNode):
            break
        if poppedNode not in visited:
            visited.append(poppedNode)
            successors = problem.getSuccessors(poppedNode)
            for successorNode in successors:
                successorNodeElement = successorNode[0]
                if successorNodeElement not in visited:
                    successorNodePath = successorNode[1]
                    successorNodeWeight = successorNode[2]
                    ## Calculating the total path covered till successor node
                    pathCoveredBySuccNode = pathTillPoppedNode + [successorNodePath]
                    ## Calculating the cumulative weight till successor node
                    cumulativeWeightTillPoppedNode += successorNodeWeight
                    ## Pushing the successor node, path covered and cumulative weight in the priority queue
                    queue.push((successorNodeElement, pathCoveredBySuccNode,cumulativeWeightTillPoppedNode),
                                cumulativeWeightTillPoppedNode)

    return pathTillPoppedNode

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    ## priority queue --> To push the nodes in the stack according to edge priority,
    ## visited --> List for storing visited nodes,
    ## Initial starting node...Push this node in priority queue along with empty directions and edge weight 0
    startNode = problem.getStartState()
    queue = util.PriorityQueue()
    visited = []
    directions = []
    queue.push((startNode, directions, 0), 0)
    pathTillPoppedNode = []
    ## 1. Pop each element out of the priority queue until it is empty
    ## 2. If the popped node is goal state, exit the loop and return the path
    ## 3. If not, check if popped node is in visited nodes. If it is not, add it to visited nodes and proceed to step 4.
    ## 4. Get the successors of popped node.
    ## 5. For each successor get the path covered and cumulative weight till the successor node.
    ## 6. Push the successor node in the priority queue along with the path, cumulative weight and heuristic function.
    ## 7. Repeat steps 1-7 until the priority queue is empty.
    while not queue.isEmpty():
        poppedNodeData = queue.pop()
        poppedNode = poppedNodeData[0]
        pathTillPoppedNode = poppedNodeData[1];
        cumulativeWeightTillPoppedNode = poppedNodeData[2];
        if problem.isGoalState(poppedNode):
            break
        if poppedNode not in visited:
            visited.append(poppedNode)
            successors = problem.getSuccessors(poppedNode)
            for successorNode in successors:
                successorNodeElement = successorNode[0]
                if successorNodeElement not in visited:
                    successorNodePath = successorNode[1]
                    successorNodeWeight = successorNode[2];
                    ## Calculating the total path covered till successor node
                    pathCoveredBySuccNode = pathTillPoppedNode + [successorNodePath]
                    ## Calculating the cumulative weight till successor node
                    cumulativeWeightTillPoppedNode += successorNodeWeight
                    ## Pushing the successor node, path covered, cumulative weight and heuristic in the priority queue
                    queue.push((successorNodeElement, pathCoveredBySuccNode,cumulativeWeightTillPoppedNode),
                                cumulativeWeightTillPoppedNode + heuristic(successorNodeElement, problem))

    return pathTillPoppedNode
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
