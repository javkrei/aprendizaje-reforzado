#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import exp, factorial
import seaborn as sns

# máximo # de autos en cada locación
MAX_CARS = 20

# máximo # de autos a mover durante la noche
 maximum # of cars to move during night
MAX_MOVE_OF_CARS = 5

# esperanza de pedidos de alquiler en la primer locación
RENTAL_REQUEST_FIRST_LOC = 3

# esperanza de pedidos de alquiler en la segunda locación
RENTAL_REQUEST_SECOND_LOC = 4

# esperanza de # de autos devueltos en la primera locación
RETURNS_FIRST_LOC = 3

# esperanza de # autos devueltos en la segunda locación
RETURNS_SECOND_LOC = 2

DISCOUNT = 0.9

# crédito ganado por un auto
RENTAL_CREDIT = 10

# costo de trasladar un auto
MOVE_CAR_COST = 2

# todas las posibles acciones
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

# límite superior de la distribución de Poisson An up bound for poisson distribution
# Si n es superior a este valor la probabilidad de obtener n es truncada a cero to 0
POISSON_UPPER_BOUND = 11

# Distribución de probabilidad Poisson
# @lam: lambda debe ser menor a 10 en este función
poisson_cache = dict()
def poisson(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = exp(-lam) * pow(lam, n) / factorial(n)
    return poisson_cache[key]

# @state: [# de autos en la primera locación, # de autos en la segunda locación]
# @action: positivo si se mueven autos de la primera locación a la segunda,
#          negativo se se mueven autos de la segunda a la primera
# @stateValue: matriz de valor de los estados (función de valor)
# @constant_returned_cars: is es True, el modelo es simplificado de tal manera 
#                          que el # de autos devueltos durante el día es constante
#                          en vez de un valor aleatorio según una distribución poisson, lo que reduce el tiempo de cálculo
#                          y deja el valor de la política/función valor casi igual
def expected_return(state, action, state_value, constant_returned_cars):
    # inicializar el retorno total
    # initailize total return
    returns = 0.0

    # costo de mover autos
    returns -= MOVE_CAR_COST * abs(action)

    # iterar sobre los posibles pedidos de alquiler
    for rental_request_first_loc in range(0, POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(0, POISSON_UPPER_BOUND):
            # moviendo los autos
            num_of_cars_first_loc = int(min(state[0] - action, MAX_CARS))
            num_of_cars_second_loc = int(min(state[1] + action, MAX_CARS))

            # pedidos de alquiler válidos deben ser menores a la cantidad de autos
            real_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
            real_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

            # obtener créditos por alquilar
            reward = (real_rental_first_loc + real_rental_second_loc) * RENTAL_CREDIT
            num_of_cars_first_loc -= real_rental_first_loc
            num_of_cars_second_loc -= real_rental_second_loc

            # probabilidad de la combinación actual de pedidos de alquiler
            prob = poisson(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * \
                         poisson(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            if constant_returned_cars:
                # obtener autos devueltos, que pueden ser usados para alguilar mañana
                returned_cars_first_loc = RETURNS_FIRST_LOC
                returned_cars_second_loc = RETURNS_SECOND_LOC
                num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])
            else:
                for returned_cars_first_loc in range(0, POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(0, POISSON_UPPER_BOUND):
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        prob_ = poisson(returned_cars_first_loc, RETURNS_FIRST_LOC) * \
                               poisson(returned_cars_second_loc, RETURNS_SECOND_LOC) * prob
                        returns += prob_ * (reward + DISCOUNT * state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])
    return returns

def figure_4_2(constant_returned_cars=True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# autos en la primera locación', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# autos en la segunda locación', fontsize=30)
        fig.set_title('política %d' % (iterations), fontsize=30)

        # evaluación de política (in-place)
        while True:
            new_value = np.copy(value)
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_value[i, j] = expected_return([i, j], policy[i, j], new_value,
                                                      constant_returned_cars)
            value_change = np.abs((new_value - value)).sum()
            print('cambio de valor %f' % (value_change))
            value = new_value
            if value_change < 1e-4:
                break

        # mejora de política
        new_policy = np.copy(policy)
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                action_returns = []
                for action in actions:
                    if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-float('inf'))
                new_policy[i, j] = actions[np.argmax(action_returns)]

        policy_change = (new_policy != policy).sum()
        print('la política cambió en %d estados' % (policy_change))
        policy = new_policy
        if policy_change == 0:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# autos en la primera locación', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# autos en la segunda locación', fontsize=30)
            fig.set_title('valor óptimo', fontsize=30)
            break

        iterations += 1

    plt.savefig('../images/figure_4_2.png')
    plt.close()

if __name__ == '__main__':
    figure_4_2()