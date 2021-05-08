import numpy as np
import matplotlib.pyplot as plt
import kmeans1d
import time
import copy
from contextlib import redirect_stdout

from typing import List
from enum import Enum
from scipy.stats import uniform


class Agent:

    def __init__(self, id, strategy):
        self.id = id
        self.strategy = strategy
        self.v = 1
        self.reports = []
        self.avarage_reputation = 0

    def __str__(self):
        return 'Id: ' + str(self.id) + ', Strategy: ' + str(self.strategy) + ', Avarage reputation: ' + str(self.avarage_reputation) + ', V: ' + str(self.v)


class AgentType(Enum):
    HONEST = 1,
    STRATEGIC = 2


class Report:
    def __init__(self, cycle, value, reported_to):
        self.cycle = cycle
        self.value = value
        self.reported_to = reported_to


class Information:
    def __init__(self, cycle, agents):
        self.cycle = cycle
        self.agents = agents

    def __str__(self):
        agentInfos = []
        for agent in self.agents:
            agentInfos.append(agent.__str__())
        return "\n######\nCycle: " + str(self.cycle) + str(", Agents: ") + "\n".join(agentInfos)


def rtbs(honest_agent_good_will, strategic_agent_good_will, stragic_agent_raport_truthfulness, percent_of_strategic_agents):
    cycles = 10
    agents = generate_agents(percent_of_strategic_agents)
    informations = []
    for cycle in range(cycles):
        print("Cycle: " + str(cycle))
        start = time.time()
        generate_reports(cycle, agents, honest_agent_good_will,
                         strategic_agent_good_will, stragic_agent_raport_truthfulness)
        rae(cycle, agents)
        agent_informations = []

        for agent in agents:
            agent_information = Agent(agent.id, agent.strategy)
            agent_information.v = agent.v
            agent_information.avarage_reputation = agent.avarage_reputation
            agent_informations.append(agent_information)

        informations.append(Information(cycle,  agent_informations))
        end = time.time()
        print('Elapsed time: ' + str(end - start))

    with open('results.txt', 'w') as f:
        for info in informations:
            with redirect_stdout(f):
                print(info)


def rae(cycle, agents):
    avarage_reputations = count_and_get_agents_avarage_reputations(
        agents, cycle)
    avarage_reputations.sort()
    clusters, centroids = kmeans1d.cluster(avarage_reputations, 2)

    for agent in agents:
        agents_similiar_values = []
        highest_values = []
        agent_index_in_avg_repurations = avarage_reputations.index(
            agent.avarage_reputation)
        agent_cluster_value = clusters[agent_index_in_avg_repurations]

        for index, avarage_reputation in enumerate(avarage_reputations):
            avg_reputation_in_cluster = clusters[index]

            if avg_reputation_in_cluster is agent_cluster_value:
                agents_similiar_values.append(avarage_reputation)

            if avg_reputation_in_cluster == 1:
                highest_values.append(avarage_reputation)

        n_i = safe_div(sum(agents_similiar_values),
                       len(agents_similiar_values))

        n_high = safe_div(sum(highest_values),
                          len(highest_values))

        v = safe_div(n_i, n_high)
        agent.v = v


def generate_agents(percent_of_strategic_agents):
    number_of_agents = 1000
    agents = []

    for index in range(int(number_of_agents * percent_of_strategic_agents)):
        agents.append(Agent(index, AgentType.STRATEGIC))

    agents_length = len(agents)

    for index in range(number_of_agents - agents_length):
        agents.append(Agent(agents_length + index, AgentType.HONEST))

    np.random.shuffle(agents)
    return agents


def generate_reports(cycle, agents, honest_agent_good_will, strategic_agent_good_will, stragic_agent_raport_truthfulness):
    for index, agent in enumerate(agents):
        providers = generate_providers(agent, agents)
        for provider in providers:
            reputation = count_reputation(
                agent, provider, honest_agent_good_will, strategic_agent_good_will, stragic_agent_raport_truthfulness)
            provider.reports.append(
                Report(cycle, reputation[0], agent))


def generate_providers(agent, agents):
    providers = np.random.choice(agents, int(np.random.uniform(5, 15, 1)))
    if agent in providers:
        generate_providers(agent, agents)
    return providers


def count_reputation(recipient, provider, honest_agent_good_will, strategic_agent_good_will, stragic_agent_raport_truthfulness):
    policy_threshold = 1
    reputation_threshold = 1

    if recipient.strategy is AgentType.STRATEGIC and provider.strategy is AgentType.HONEST:
        policy_threshold = strategic_agent_good_will

    if recipient.strategy is AgentType.HONEST and provider.strategy is AgentType.STRATEGIC:
        reputation_threshold = stragic_agent_raport_truthfulness

    if recipient.strategy is AgentType.HONEST:
        policy_threshold = l(recipient.v, honest_agent_good_will)

    if provider.strategy is AgentType.HONEST:
        reputation_threshold = l(provider.v, honest_agent_good_will)

    services_availability = np.random.uniform(0, 1, 1)
    behaviour_policy = np.minimum(services_availability, policy_threshold)

    return np.minimum(behaviour_policy, reputation_threshold)


def l(v, x):
    return 1 if v >= 1-x else 0


def count_and_get_agents_avarage_reputations(agents, cycle):
    reputations = []
    for agent in agents:
        avarage_reputation = 0
        for report in agent.reports:
            if report.cycle <= cycle:
                avarage_reputation += report.reported_to.v * \
                    report.value
        agent.avarage_reputation = avarage_reputation
        reputations.append(agent.avarage_reputation)
    return reputations


def safe_div(x, y):
    return 0 if y == 0 else x / y


###### Start RTBS ##############################
rtbs(0.5, 0.5, 0.5, 0.4)
