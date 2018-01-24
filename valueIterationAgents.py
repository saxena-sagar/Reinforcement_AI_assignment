# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
from matplotlib.cbook import Null
from numpy.core.fromnumeric import argmax

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"  
        for i in range(self.iterations):#iterating through all possible states
            values_updated = util.Counter()#creating an empty list to hold the best action for each state
            #Updating new value k+1 for all the states
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    #checking if terminal state then value need not be updated and next state checked for
                    continue
                best_action = self.computeActionFromValues(state)
                values_updated[state] = self.computeQValueFromValues(state, best_action)
            self.values = values_updated
  
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_val = 0
        
        #getting new next state,probability  from getTransitionState
        for new_state, probabilities in self.mdp.getTransitionStatesAndProbs(state, action):
            sum_over_rewards = self.mdp.getReward(state,action,new_state) + self.discount * self.getValue(new_state)
            Q_val += probabilities * sum_over_rewards
        return Q_val 
            
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.mdp.getPossibleActions(state)
        action_val = []
        
        #checking if no legal actions
        if not legal_actions:
            return None
        action_max = None
        maxQ = -9999#To  compare and get the best action corresponding to the maximum value from  a state
        for action in legal_actions:
            if self.computeQValueFromValues(state, action) > maxQ:
                maxQ = self.computeQValueFromValues(state, action)
                action_max = action
        return action_max        
         
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
