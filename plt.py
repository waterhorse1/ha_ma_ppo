
import csv
from tensorboard.backend.event_processing import event_accumulator
import json
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + './')


def read_tfevent(event,target):
    ea = event_accumulator.EventAccumulator(event)
    ea.Reload()
    episode_rewards = ea.scalars.Items(target)
    episode_reward = [[i.step, i.value] for i in episode_rewards]
    # episode_reward = [[i.step/1e5, i.value] for i in episode_rewards]
    return episode_reward

def read_tfevent_lists(path, all_files, target):
    # all_files=[]
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            all_files=read_tfevent_lists(cur_path,all_files,target)
        else:
            if path.find(target)!=-1 or file.find(target)!=-1:
                all_files.append( path + "/" + file)
    return all_files

def return_data(path,target):
    tfevents = read_tfevent_lists(path, [], target)
    episode_rewards = []
    for event in tfevents:
        episode_reward = read_tfevent(event,target)  # 10000 * 2
        episode_rewards.append(episode_reward)  # 10 * 10000 * 2
    return episode_rewards

def read_csv_2_dict(csv_str, step=1):
    csv_reader = csv.reader(open(csv_str))
    i = 0
    list_all = []
    for row in csv_reader:
        if i > 0 and i % step == 0:
            list_all.append([int(row[1]), float(row[2])])
        i += 1
    return list_all


def smoothen(data, w):
    res = np.zeros_like(data)
    for i in range(len(data)):
        if i > w:
            res[i] = np.mean(data[i-w:i])
        elif i > 0:
            res[i] = np.mean(data[:i])
        else:  # i == 0
            res[i] = data[i]
    return res


def draw(data_dict, i, w):
    color = ['orange', 'hotpink', 'dodgerblue', 'mediumpurple', 'c', 'cadetblue', 'steelblue', 'mediumslateblue',
             'hotpink', 'mediumturquoise']
    plt.xlabel("Environment steps", fontsize=18)
    plt.ylabel("Average Episode Reward", fontsize=18)

    for k, episode_rewards in data_dict.items():
        timestep = np.array(episode_rewards)[:, :, 0][0]
        if scenario == "HumanoidStandup":
            reward = np.array(episode_rewards)[:, :, 1] / 1000
        else:
            reward = np.array(episode_rewards)[:, :, 1]
        r_mean, r_std = np.mean(reward, axis=0), np.std(reward, axis=0, ddof=1)

        r_mean = smoothen(r_mean, w)
        r_std = smoothen(r_std, w)

        plt.plot(timestep, r_mean, color=color[i], label=k, linewidth=1.5)
        plt.fill_between(timestep, r_mean - r_std, r_mean + r_std, alpha=0.2, color=color[i])
        i += 1


if __name__ == "__main__":
    # scenario = "Ant"
    # config = "2x4"
    smoothen_w = 5
    need_legend = True
    # need_legend = True
    # if scenario == "Reacher":
    #     plt.ylim(ymin=-200, ymax=0)
    # if scenario == "ManyAgentSwimmer":
    #     plt.ylim(ymin=-150, ymax=250)
    # if scenario == "HumanoidStandup":
    #     plt.text(-1e6, 145, r'1e3', fontsize=10)

    data_dict = {}
    # task=["ant2x4","ant8x1","walker2x3","walker6x1","cheetah2x3","cheetah6x1"]
    scenario="Walker"
    config="6x1"
    # algo=["norandom","share","ori"]
    episode_rewards = []
    path="C:\\Users\\MeteorRain\\Desktop\\ablation\\"+scenario.lower()+config+"\\ori"
    episode_rewards=return_data(path,"eval_average_episode_rewards")
    data_dict['HATRPO (original)'] = episode_rewards

    episode_rewards = []
    path="C:\\Users\\MeteorRain\\Desktop\\ablation\\"+scenario.lower()+config+"\\share"
    episode_rewards=return_data(path,"eval_average_episode_rewards")
    data_dict['HATRPO (shared parameter)'] = episode_rewards

    episode_rewards = []
    path="C:\\Users\\MeteorRain\\Desktop\\ablation\\"+scenario.lower()+config+"\\norandom"
    episode_rewards=return_data(path,"eval_average_episode_rewards")
    data_dict['HATRPO (no random order)'] = episode_rewards
    # episode_rewards = []
    # for i in range(5):
    #     index = i + 1
    #     csv_path = "../../../csv/" + scenario + "/" + scenario + config + "/sadppo/" + str(index) + ".csv"
    #     list_ = read_csv_2_dict(csv_path)
    #     episode_rewards.append(list_)
    # data_dict['HAPPO (ours)'] = episode_rewards

    episode_rewards = []
    for i in range(5):
        index = i + 1
        path="C:\\Users\\MeteorRain\\Desktop\\ablation\\"+scenario.lower()+config+"\\mappo_share\\"+ str(index) + ".csv"
        list_ = read_csv_2_dict(path)
        episode_rewards.append(list_)
    data_dict['MAPPO'] = episode_rewards

    # episode_rewards = []
    # for i in range(5):
    #     index = i + 1
    #     csv_path = "../../../csv/" + scenario + "/" + scenario + config + "/ippo/" + str(index) + ".csv"
    #     list_ = read_csv_2_dict(csv_path)
    #     episode_rewards.append(list_)
    # data_dict['IPPO'] = episode_rewards

    # episode_rewards = []
    # for i in range(3):
    #     index = i + 1
    #     csv_path = "../../../csv/" + scenario + "/" + scenario + config + "/maddpg/" + str(index) + ".csv"
    #     list_ = read_csv_2_dict(csv_path, 2)
    #     episode_rewards.append(list_)
    # data_dict['MADDPG'] = episode_rewards



    draw(data_dict, 0, smoothen_w)

    plt.title(scenario + " " + config, fontsize=18, pad=12)
    if need_legend:
        plt.legend(loc="upper left", fontsize=10)
        
        # plt.legend(loc="lower right", fontsize=14)
    plt.grid()
    if need_legend:
        save_path = '' + scenario + config + '.pdf'
    else:
        save_path = '' + scenario + config + '.pdf'
    plt.savefig(save_path, format='pdf')
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    plt.show()
