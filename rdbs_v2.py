import numpy as np
import matplotlib.pyplot as plt
import kmeans1d
import time

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


def rtbs(honest_agent_good_will, strategic_agent_good_will, stragic_agent_raport_truthfulness, percent_of_strategic_agents):
    cycles = 100
    agents = generate_agents(percent_of_strategic_agents)

    for cycle in range(cycles):
        print("Cycle: " + str(cycle))
        start = time.time()
        generate_reports(cycle, agents, honest_agent_good_will,
                         strategic_agent_good_will, stragic_agent_raport_truthfulness)
        rae(cycle, agents)
        end = time.time()
        print('Elapsed time: ' + str(end - start))

    for agent in agents:
        print(agent)


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

            if avg_reputation_in_cluster is 1:
                highest_values.append(avarage_reputation)

        agent.v = (sum(agents_similiar_values) / len(agents_similiar_values)
                   ) / (sum(highest_values) / len(highest_values))


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
            provider_index = agents.index(provider)
            agents[provider_index].reports.append(
                Report(cycle, reputation, agent))


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

    if recipient.strategy is AgentType.HONEST and provider.strategy is AgentType.HONEST:
        policy_threshold = 1 - honest_agent_good_will
        reputation_threshold = 1 - honest_agent_good_will

    if recipient.strategy is AgentType.HONEST and provider.strategy is AgentType.STRATEGIC:
        reputation_threshold = stragic_agent_raport_truthfulness

    services_availability = np.random.uniform(0, 1, 1)
    behaviour_policy = np.minimum(services_availability, policy_threshold)

    return np.minimum(behaviour_policy, reputation_threshold)


def count_and_get_agents_avarage_reputations(agents, cycle):
    reputations = []
    for agent in agents:
        avarage_reputation = 0
        for report in agent.reports:
            if report.cycle > cycle:
                continue
            reported_to_index = agents.index(report.reported_to)
            avarage_reputation += agents[reported_to_index].v * report.value
        agent.avarage_reputation = avarage_reputation
        reputations.append(agent.avarage_reputation)
    return reputations


###### Start RTBS ##############################
rtbs(0.5, 0.5, 0.5, 0.4)
