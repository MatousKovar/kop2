import math
import random
import numpy as np
from MWSATInstance import MWSATInstance
from MWSATSolution import MWSATSolution

def compare_states(lhs: MWSATSolution, rhs: MWSATSolution):
    """returns 1 if lhs score is bigger than rhs, 0 if other """
    if lhs.clauses_satisfied > rhs.clauses_satisfied:
        return 1
    if rhs.clauses_satisfied > lhs.clauses_satisfied:
        return 0
    else:
        return lhs.current_score > rhs.current_score


def simulated_annealing(instance: MWSATInstance, 
                        P0: float, 
                        cooling_coefficient: float, 
                        equilibrium_steps: int, 
                        max_steps_without_improvement: int,
                        fitness_coefficient: float = None):
    """
    Simulated Annealing Solver for MWSAT.
    
    Hyperparameters:
    - initial_temperature (T0): Starting temperature. Controls willingness to accept bad moves.
    - cooling_coefficient (alpha): How fast T decreases (usually 0.8 - 0.99). 
    - equilibrium_steps (Inner Loop): Steps performed at each temperature level. [cite: 346]
    - min_temperature: Stopping condition (Frozen). [cite: 316]
    """
    
    current_state = MWSATSolution(instance)
    best_state = current_state.copy()

    # setting initial temperature
    delta_avg = set_delta(instance,1e6,cooling_coefficient = 1,equilibrium_steps=100,steps=3000)

    temperature = abs(delta_avg) / abs(np.log(P0))

    history = []


    steps_without_improvement = 0
    while steps_without_improvement < max_steps_without_improvement:
        
        # first parameter in whitebox phase - how many steps in equilibrium?
        for _ in range(equilibrium_steps):
            steps_without_improvement += 1
            
            neighbor = current_state.generate_neighbor()

            # delta is how better or worse the solution is in terms of clauses satisfied combined with normalized sum of weights
            if fitness_coefficient:
                normalized_score = (neighbor.current_score - current_state.current_score) / instance.max_weight
                delta = (1-fitness_coefficient) * (neighbor.clauses_satisfied - current_state.clauses_satisfied) + fitness_coefficient * normalized_score
            else:
                delta = neighbor.clauses_satisfied - current_state.clauses_satisfied

            # delta should fallback to score if clauses satisfied is same
            # if delta == 0:
            #     
            
            if compare_states(neighbor,current_state):
                current_state = neighbor
                if compare_states(current_state,best_state):
                    steps_without_improvement = 0
                    best_state = current_state.copy()
                        
            else:
                probability = math.exp(delta / temperature)
                if random.random() < probability:
                    current_state = neighbor # Accept worsening (Diversification)
            
            # log current score
            history.append(neighbor.current_score)
            # history_satisfied.append(current_state.clauses_satisfied)
            # history_score.append(current_state.get_raw_score())

        
        temperature *= cooling_coefficient
        
    return best_state, history


def set_delta(instance: MWSATInstance, 
              initial_temperature: float, 
              cooling_coefficient: float, 
              equilibrium_steps: int, 
              steps: int,
              fitness_coefficient: float = None):
    current_state = MWSATSolution(instance)
    best_state = current_state.copy()
    temperature = initial_temperature

    deltas = []

    step_counter = 0
    while step_counter < steps:
        
        for _ in range(equilibrium_steps):
            step_counter += 1
            
            neighbor = current_state.generate_neighbor()

            # delta is how better or worse the solution is in terms of clauses satisfied
            if fitness_coefficient:
                normalized_score = (neighbor.current_score - current_state.current_score) / instance.max_weight
                delta = (1-fitness_coefficient) * (neighbor.clauses_satisfied - current_state.clauses_satisfied) + fitness_coefficient * normalized_score
            else:
                delta = neighbor.clauses_satisfied - current_state.clauses_satisfied            
            # if delta == 0:
            #     delta = (neighbor.current_score - current_state.current_score) / instance.max_weight

            if compare_states(neighbor,current_state):
                current_state = neighbor
                if compare_states(current_state,best_state):
                    best_state = current_state.copy()
                        
            else:

                probability = math.exp(delta / temperature)
                if random.random() < probability:
                    deltas.append(delta)
                    current_state = neighbor # Accept worsening (Diversification)
            
            # log current score
            
            # history_satisfied.append(current_state.clauses_satisfied)
            # history_score.append(current_state.get_raw_score())

        
        temperature *= cooling_coefficient
        
    return np.mean(deltas)