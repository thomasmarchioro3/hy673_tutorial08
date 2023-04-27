import numpy as np
import matplotlib.pyplot as plt

def get_transition_matrix(p=0.5, q=0.5):
    # define the transition probability matrix
    return np.asarray(
        [
            [1-p, p, 0, 0, 0, 0],  # state 1
            [1-q, 0, q, 0, 0, 0],  # state 2
            [0, 1-p, 0, p, 0, 0],  # state 3
            [0, 0, 1-p, 0, p, 0],  # state 4
            [0, 0, 0, 1-p, 0, p],  # state 5
            [0, 0, 0, 0, 1-p, p],  # state 6
        ]
    )

"""
Simulate a random walk
Inputs:
- P: transition probability matrix (describes the Markov chain).
- n_steps: the number of steps in the random walk.
"""
def random_walk(P, n_steps=1000):

    # we enumerate them from 0 to 5 for simplicity
    states = np.arange(len(P))
    curr_state = 0  # choice of the initial state

    counts = np.zeros(len(P))  # initialize state counters
    
    for t in range(n_steps):
        prev_state = curr_state
        curr_state = np.random.choice(states, p=P[prev_state])  # choose the current state based on P and on the previous state
        counts[curr_state] += 1  # update the counter of the current state
    
    probs = counts / n_steps
    return probs

"""
Get stationary probabilities based on the eigenvector of the eigenvalue 1.
"""
def get_stationary_probs(P):

    eigenvals, eigenvec = np.linalg.eig(P.T)  # eigenvalue decomposition of P transposed
    i_1, = np.argwhere(np.abs(eigenvals - 1) < 1e-4)  # index for eigenvalue 1 ("close" to 1 to account for computational errors)
    probs, = eigenvec[:, i_1].T  # select eigenvector correspondent to eigenvalue 1
    probs = probs / probs.sum()  # normalize
    return probs

"""
Prints stationary probabilities (according to the eigenvalue decomposition)
"""
def solution(p, q, ex='a', scale='linear'):
    P = get_transition_matrix(p, q)
    probs_theory = get_stationary_probs(P)
    print(f"Stationary probabilities for ex 4{ex}:", probs_theory)

    for T in [1000, 100_000]:
        probs_experiment = random_walk(P, n_steps=T)
        plt.figure()
        plt.bar(np.arange(1, len(P)+1, 1)-0.2, probs_experiment, width=0.4, color='tab:blue', label='random walk')
        plt.bar(np.arange(1, len(P)+1, 1)+0.2, probs_theory, width=0.4, color='tab:pink', label='eigenvalue decomposition')
        if scale == 'log':
            plt.yscale('log')
        plt.xlabel('State')
        plt.ylabel('Probability')
        # plt.grid(linestyle=':')
        plt.legend()
        plt.title(f"Exercise 4{ex}, T=10^{int(np.log10(T))}")
        plt.savefig(f'results/ex4{ex}_T1e{int(np.log10(T))}_{scale}.eps')
        plt.draw()

if __name__ == "__main__":

    np.random.seed(42)  # set random seed for reproducibility

    solution(p=0.5, q=0.5, ex='b+c', scale='linear')
    solution(p=0.5, q=0.1, ex='d', scale='linear')
    solution(p=0.1, q=0.1, ex='e', scale='log')

    plt.show()
    

