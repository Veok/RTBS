# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Importing
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import kmeans1d
import time
from typing import List

# %%
# Inputs
x = 0.5
y = 0.5
z = 0.5


# %%
# Variables
N = 1000  # number of agents
t = 100  # cycles
ksi = 0.4  # percent of strategic agents in range from 0 to 50%


# %%
# Agents

class Agent:
    def __init__(self, id, strategy, credibility, previousCredibility):
        self.id = id
        self.strategy = strategy
        self.credibility = credibility
        self.previousCredibility = previousCredibility

    def __str__(self):
        return 'Id: ' + str(self.id) + ', Strategy: ' + str(self.strategy) + ', Credibility: ' + str(self.credibility)

####################################


agents = []

for index in range(int(N * ksi)):
    agents.append(Agent(index, 'S', 1, 0))

agentsLength = len(agents)

for index in range(N - agentsLength):
    agents.append(Agent(agentsLength + index, 'H', 1, 0))

np.random.shuffle(agents)


# %%
# Choosing service suppliers
for cycle in range(t):
    print('Cycle: ' + str(cycle))
    start = time.time()
    for index, agent in enumerate(agents):
        setOfServiceSuppliersUniform = np.random.uniform(5, 15, 1)
        setOfServiceSuppliers = np.random.choice(
            agents, int(setOfServiceSuppliersUniform))

        summarizedReputation = 0
        for serviceSupplier in setOfServiceSuppliers:
            p = 1
            r = 1

            if agent.strategy == 'S' and serviceSupplier.strategy == 'S':
                p = 1
                r = 1

            if agent.strategy == 'S' and serviceSupplier.strategy == 'H':
                p = y

            if agent.strategy == 'H' and serviceSupplier.strategy == 'H':
                p = 1 - x
                r = 1 - x

            if agent.strategy == 'H' and serviceSupplier.strategy == 'S':
                r = z

            aij = np.random.uniform(0, 1, 1)
            pt = np.minimum(aij, p)
            rt = np.minimum(pt, r)
            serviceSupplier.previousCredibility = rt
            # summarizedReputation += (serviceSupplier.credibility * rt)

        # agent.credibility = summarizedReputation[0]
        agentsCredibility = []

        for agentCreds in agents:
            sum = 0
            for agents_j in agents:
                if agents_j.id == agentCreds.id:
                    continue
                sum += agents_j.credibility * agentCreds.previousCredibility

            agentsCredibility.append(sum)
            # agentsCredibility.append(agentCreds.previousCredibility)

        agentsCredibility.sort()
        clusters, centroids = kmeans1d.cluster(agentsCredibility, 2)

        agentIndexInCreds = agentsCredibility.index(agent.credibility)
        agentSetValue = clusters[agentIndexInCreds]
        indexesOfSimilar = []
        indexesOfOpposites = []
        similarValues = []
        oppositeValues = []

        for index, element in enumerate(agentsCredibility):
            agentCredoSetValue = clusters[index]
            if agentCredoSetValue is agentSetValue:
                similarValues.append(element)
            if agentCredoSetValue is 1:
                oppositeValues.append(element)

        agentsAvarage = (sum(similarValues) / len(similarValues)) / \
            (sum(oppositeValues) / len(oppositeValues))
        agent.credibility = agentsAvarage

    end = time.time()
    print('Elapsed time: ' + str(end - start))

for agent in agents:
    print(agent)


# %%
