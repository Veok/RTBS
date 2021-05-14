import numpy as np
import matplotlib.pyplot as plt
import kmeans1d
import time
import copy
import pandas as pd

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
    HONEST = 1
    STRATEGIC = 2


class Report:
    def __init__(self, cycle, value, reported_to):
        self.cycle = cycle
        self.value = value
        self.reported_to = reported_to


def rtbs(honest_agent_good_will, strategic_agent_good_will, stragic_agent_raport_truthfulness, percent_of_strategic_agents):
    cycles = 100
    agents = generate_agents(percent_of_strategic_agents)
    infos = []
    for cycle in range(cycles):
        print("Cycle: {} ".format(cycle))

        start = time.time()
        generate_reports(cycle, agents, honest_agent_good_will,
                         strategic_agent_good_will, stragic_agent_raport_truthfulness)
        rae(cycle, agents)
        save_cycle_information(infos, agents, cycle)

        end = time.time()
        print('Elapsed time: ' + str(end - start))

    file_name = 'results_x' + str(honest_agent_good_will) + '_y' + str(
        strategic_agent_good_will) + '_z' + str(stragic_agent_raport_truthfulness)

    save_data(infos, file_name)


def save_cycle_information(infos, agents, cycle):
    strategic_agents = []
    honest_agents = []

    for agent in agents:
        if agent.strategy == AgentType.STRATEGIC:
            strategic_agents.append(agent.v)
        else:
            honest_agents.append(agent.v)

    v_mean_strategic = np.mean(strategic_agents)
    v_mean_honest = np.mean(honest_agents)

    infos.append([cycle,  v_mean_strategic, v_mean_honest])


def save_data(infos, file_name):
    df = pd.DataFrame(
        infos, columns=["Cycle", "Honest Agent", "Strategic Agent"])
    ax = df.plot(x="Cycle", title="Agents in cycles")
    fig = ax.get_figure()
    fig.savefig(file_name + '.svg')
    df.to_json(path_or_buf=file_name + '.json')


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

        sum_n_i = safe_div(sum(agents_similiar_values),
                           len(agents_similiar_values))

        sum_n_high = safe_div(sum(highest_values),
                              len(highest_values))

        v = safe_div(sum_n_i, sum_n_high)
        agent.v = v


def generate_agents(percent_of_strategic_agents):
    number_of_agents = 10
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
# rtbs(x, y, z, ksi)
rtbs(0.5, 0.5, 0.5, 0.4)
rtbs(0.7, 0.2, 0.2, 0.4)
rtbs(0.2, 0.7, 0.2, 0.4)
rtbs(0.2, 0.2, 0.7, 0.4)
rtbs(0.7, 0.7, 0.2, 0.4)
rtbs(0.7, 0.7, 0.7, 0.4)
rtbs(0.2, 0.7, 0.7, 0.4)
rtbs(0.4, 0.8, 0.7, 0.4)
rtbs(0.8, 0.4, 0.2, 0.4)
rtbs(0.2, 0.8, 0.4, 0.4)
rtbs(0.9, 0.1, 0.9, 0.4)

rtbs(0.5, 0.5, 0.5, 0.2)
rtbs(0.7, 0.2, 0.2, 0.2)
rtbs(0.2, 0.7, 0.2, 0.2)
rtbs(0.2, 0.2, 0.7, 0.2)
rtbs(0.7, 0.7, 0.2, 0.2)
rtbs(0.7, 0.7, 0.7, 0.2)
rtbs(0.2, 0.7, 0.7, 0.2)
rtbs(0.4, 0.8, 0.7, 0.2)
rtbs(0.8, 0.4, 0.2, 0.2)
rtbs(0.2, 0.8, 0.4, 0.2)
rtbs(0.9, 0.1, 0.9, 0.2)
