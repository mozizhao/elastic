import copy
import json
import random
import _thread
import threading
import time

import networkx
import logging
import requests


def arrive(t):
    global job_idx
    result = {}
    with open("trace.dat", "r") as f:
        for l in list(filter((lambda line: line.split(',')[0] == str(t)), f.readlines())):
            result[f'job-{job_idx}'] = {
                'arrival': t, 'job_id': f'job-{job_idx}', 'job_type': l.split(',')[1],
                'iteration': int(l.split(',')[2]), 'demand': int(l.split(',')[3]), 'batch_size': int(l.split(',')[4]),
                'progress': 0., 'prev_host': None, 'prev_workers': None
            }
            job_idx += 1
    return result


def schedule():
    # bipartite matching based method, 20230106
    global all_job_limit, available_jobs

    if len(available_jobs) <= 0:
        return {}

    # first, find bipartite matching between single-worker jobs and gpus
    graph = networkx.Graph()
    for gpu_type in gpu_types:
        for i in range(gpu_nums[gpu_type] // int(all_job_limit)):
            graph.add_node(f'{gpu_type}_{i}', bipartite=0)
    for job_id in available_jobs:
        graph.add_node(job_id, bipartite=1)

    for gpu_type in gpu_types:
        for i in range(gpu_nums[gpu_type] // all_job_limit):
            for job_id in available_jobs:
                j_attr = available_jobs[job_id]
                iter, job_type, batch_size = j_attr['iteration'] - j_attr['progress'], j_attr['job_type'], j_attr[
                    'batch_size']
                weight = iter * (cal_iteration_time(gpu_type, job_type, batch_size,
                                                    all_job_limit) ** 2) / cal_iteration_time(gpu_type, job_type,
                                                                                              batch_size, 1)

                graph.add_edge(f'{gpu_type}_{i}', job_id, weight=weight)

    raw_single_matching = networkx.bipartite.minimum_weight_full_matching(graph)
    worker_nums, matching = {}, {}

    for k, v in raw_single_matching.items():
        if '_' in k and k.split('_')[0] in gpu_types:
            gpu_type, jid = k.split('_')[0], v
            if not matching.get(gpu_type):
                matching[gpu_type] = []
            matching[gpu_type].append(jid)
            worker_nums[jid] = all_job_limit

    # list all scheduled jobs
    scheduled_jobs = []
    for type in gpu_types:
        if matching.get(type):
            scheduled_jobs.extend(matching[type])

    # iterates GPUs from high-end to low-end
    for gpu_type in gpu_types:
        if not matching.get(gpu_type):
            continue

        shrink_slots = 0

        for j in matching[gpu_type]:
            # if shrinkable jobs exist, shrink one of its GPU.
            while worker_nums[j] > 1 and \
                    iteration_time[gpu_type][available_jobs[j]['job_type']][available_jobs[j]['batch_size']][
                        worker_nums[j]] > \
                    iteration_time[gpu_type][available_jobs[j]['job_type']][available_jobs[j]['batch_size']][
                        worker_nums[j] - 1]:
                worker_nums[j] -= 1
                shrink_slots += 1

        # pick jobs that can benefit most from one GPU (progress most from this single GPU)
        for _ in range(shrink_slots):
            unsched_jobs = sorted(
                [jid for jid in available_jobs if jid not in scheduled_jobs],
                # key=lambda j: cal_throughput(gpu_type, available_jobs[j]['job_type'], available_jobs[j]['batch_size'], 1) / (available_jobs[j]['iteration'] - available_jobs[j]['progress']),
                key=lambda j: cal_throughput(gpu_type, available_jobs[j]['job_type'], available_jobs[j]['batch_size'],
                                             1) / available_jobs[j]['iteration'],
                reverse=True
            )

            if not unsched_jobs:
                break

            unsched_jid = unsched_jobs[0]

            worker_nums[unsched_jid] = 1
            scheduled_jobs.append(unsched_jid)
            matching[gpu_type].append(unsched_jid)

        # turn over GPUs from low-efficiency job to high-efficiency job
        while True:
            # evaluate how the JCT of jobs degrades if one GPU is deducted.
            sched_jobs = sorted(
                list(filter(lambda j: worker_nums[j] > 1, matching.get(gpu_type))),
                # key=lambda j: (cal_throughput(gpu_type, available_jobs[j]['job_type'], available_jobs[j]['batch_size'], worker_nums[j]) - cal_throughput(gpu_type, available_jobs[j]['job_type'], available_jobs[j]['batch_size'], worker_nums[j] - 1)) / (available_jobs[j]['iteration'] - available_jobs[j]['progress']),
                key=lambda j: (cal_throughput(gpu_type, available_jobs[j]['job_type'], available_jobs[j]['batch_size'],
                                              worker_nums[j]) - cal_throughput(gpu_type, available_jobs[j]['job_type'],
                                                                               available_jobs[j]['batch_size'],
                                                                               worker_nums[j] - 1)) / available_jobs[j][
                                  'iteration'],
                reverse=False
            )

            if not sched_jobs:
                break
            # pick one job that is already selected, with maximum price descend by removing one worker
            jid = sched_jobs[0]
            # maximum descending price from jid
            # asc_jct = (cal_throughput(gpu_type, available_jobs[jid]['job_type'], available_jobs[jid]['batch_size'], worker_nums[jid]) - cal_throughput(gpu_type, available_jobs[jid]['job_type'], available_jobs[jid]['batch_size'], worker_nums[jid] - 1)) / (available_jobs[jid]['iteration'] - available_jobs[jid]['progress'])
            asc_jct = (cal_throughput(gpu_type, available_jobs[jid]['job_type'], available_jobs[jid]['batch_size'],
                                      worker_nums[jid]) - cal_throughput(gpu_type, available_jobs[jid]['job_type'],
                                                                         available_jobs[jid]['batch_size'],
                                                                         worker_nums[jid] - 1)) / available_jobs[jid][
                          'iteration']

            unsched_jobs = [j for j in available_jobs if j not in scheduled_jobs]

            if not unsched_jobs:
                break
            # replace the jid by the job with minimum price given one worker
            unsched_jid = sorted(
                unsched_jobs,
                key=lambda j: cal_throughput(gpu_type, available_jobs[j]['job_type'], available_jobs[j]['batch_size'],
                                             1) / (available_jobs[j]['iteration'] - available_jobs[j]['progress']),
                reverse=True
            )[0]
            # single-worker price of unsched_jid
            # desc_jct = cal_throughput(gpu_type, available_jobs[unsched_jid]['job_type'], available_jobs[unsched_jid]['batch_size'], 1) / (available_jobs[unsched_jid]['iteration'] - available_jobs[unsched_jid]['progress'])
            desc_jct = cal_throughput(gpu_type, available_jobs[unsched_jid]['job_type'],
                                      available_jobs[unsched_jid]['batch_size'], 1) / available_jobs[unsched_jid][
                           'iteration']

            if asc_jct < desc_jct:

                worker_nums[jid] -= 1
                worker_nums[unsched_jid] = 1
                scheduled_jobs.append(unsched_jid)
                matching[gpu_type].append(unsched_jid)
            else:
                break

    result = {}

    for gpu_type in gpu_types:
        if not matching.get(gpu_type):
            continue

        for jid in matching[gpu_type]:
            assert worker_nums[jid] != 0

            result[jid] = {'type': gpu_type, 'job_type': available_jobs[jid]['job_type'], 'num': worker_nums[jid],
                           'bs': available_jobs[jid]['batch_size']}

    return result


# calculates the iteration time of iteration time or the fitted iteration time, depending on the user's choice.
def cal_iteration_time(gpu_type, job_type, batch_size, gpu_count):
    return iteration_time[gpu_type][job_type][batch_size][gpu_count] / round_len


def cal_throughput(gpu_type, job_type, batch_size, gpu_count):
    return 1 / cal_iteration_time(gpu_type, job_type, batch_size, gpu_count)


def allocate(plan, prev_plan):
    # the first two record the data where hosts are keys, while the last two record the data where jobs are keys
    global t, prev_host_result, prev_host_worker_result, prev_jobs_location
    # record host information and worker information respectively in this round
    host_result, host_worker_result = {}, {}

    # process jobs on type by type
    for gpu_type in gpu_types:
        # list of used hosts
        used_host_list = []
        # multi-worker jobs placed on this gpu type in this time
        multi_worker_jobs = list(filter(lambda j: plan[j]['num'] > 1 and plan[j]['type'] == gpu_type, plan.keys()))
        # jobs that also have multi (or single) worker in last time slot, indicating elasticity can come into effective
        multi_worker_jobs_exist_prev = list(
            filter(lambda j: prev_plan.get(j) and prev_plan[j]['type'] == gpu_type and prev_plan[j]['num'] >= 1,
                   multi_worker_jobs))

        # first step: schedule these elastic-able jobs on their original host
        for j in multi_worker_jobs_exist_prev:
            # if some multi-worker jobs have occupied this host, then randomly select another one
            prev_located_host, selected_host = prev_jobs_location[j], None
            if prev_located_host in used_host_list:
                selected_host = random.choice(list(set(hosts_of_each_gpu_type[gpu_type]) - set(used_host_list)))
            else:
                selected_host = prev_located_host

            # in case records for this host are not initiated
            if not host_worker_result.get(selected_host):
                host_result[selected_host] = []
                host_worker_result[selected_host] = {}

            # for each multi-worker jobs should occupy the preceding workers as possible
            for i in range(plan[j]['num']):
                host_worker_result[selected_host][i] = j
            host_result[selected_host].append(j)
            used_host_list.append(selected_host)

        # second step: jobs that also have multi (or single) worker but not exist in last time slot
        for j in [j for j in multi_worker_jobs if j not in multi_worker_jobs_exist_prev]:
            selected_host = random.choice(list(set(hosts_of_each_gpu_type[gpu_type]) - set(used_host_list)))
            if not host_worker_result.get(selected_host):
                host_result[selected_host] = []
                host_worker_result[selected_host] = {}
            # for each multi-worker jobs should occupy the preceding workers as possible
            for i in range(plan[j]['num']):
                host_worker_result[selected_host][i] = j
            host_result[selected_host].append(j)
            used_host_list.append(selected_host)

        # prepared for single-worker jobs
        available_slots_in_hosts = {}
        for host in hosts_of_each_gpu_type[gpu_type]:
            available_slots_in_hosts[host] = gpus_each_host - (
                0 if not host_worker_result.get(host) else len(host_worker_result[host]))

        # single-worker jobs in this time
        single_worker_jobs = list(filter(lambda j: plan[j]['num'] == 1 and plan[j]['type'] == gpu_type, plan.keys()))
        for j in single_worker_jobs:
            # if this single worker job is scheduled in this gpu type in last time slot
            if prev_plan.get(j) and (prev_jobs_location[j] in hosts_of_each_gpu_type[gpu_type]) and (
                    available_slots_in_hosts[prev_jobs_location[j]] > 0):
                # assert that at least one host is available for jobs
                assert len([h for h in hosts_of_each_gpu_type[gpu_type] if available_slots_in_hosts[h] > 0]) > 0

                selected_host = prev_jobs_location[j]
                if not host_worker_result.get(selected_host):
                    host_worker_result[selected_host] = {}
                    host_result[selected_host] = []
                host_worker_result[selected_host][len(host_worker_result.get(selected_host))] = j
                host_result[selected_host].append(j)
                available_slots_in_hosts[selected_host] -= 1

            else:
                # assert that at least one host is available for jobs
                assert len([h for h in hosts_of_each_gpu_type[gpu_type] if available_slots_in_hosts[h] > 0]) > 0

                selected_host = random.choice(
                    [h for h in hosts_of_each_gpu_type[gpu_type] if available_slots_in_hosts[h] > 0])
                # selected_worker = len(host_worker_result[selected_host]) if host_worker_result.get(selected_host) else 0
                # select the min-numbered worker from selected host
                if not host_worker_result.get(selected_host):
                    host_result[selected_host] = []
                    host_worker_result[selected_host] = {}

                selected_worker = len(host_worker_result.get(selected_host))
                host_result[selected_host].append(j)
                host_worker_result[selected_host][selected_worker] = j
                available_slots_in_hosts[selected_host] -= 1

    # start control plane
    logging.info(f"t={t}, placement host_worker_result={host_worker_result}")
    proceed_placement(host_worker_result, prev_host_worker_result)

    # resume three global variables: prev_host_result, prev_host_worker_result, prev_jobs_location

    # clear prev_* variables
    prev_jobs_location.clear()
    prev_host_result.clear()
    prev_host_worker_result.clear()

    for gpu_type in gpu_types:
        for j in [j for j in plan if plan[j]['type'] == gpu_type]:
            assert len([k for (k, v) in host_result.items() if j in v]) > 0
            host = [k for (k, v) in host_result.items() if j in v][0]
            prev_jobs_location[j] = host

    prev_host_result = copy.deepcopy(host_result)
    prev_host_worker_result = copy.deepcopy(host_worker_result)
    logging.info(f'After completing round {t}, the global variables are: prev_host_result={prev_host_result}, '
                 f'prev_host_worker_result={prev_host_worker_result}, and prev_jobs_location={prev_jobs_location}')


def stop(related_old_jobs, host):
    logging.info(f"send stop request with params: jobs={related_old_jobs}")
    ret = requests.post('http://' + host + ':' + str(CLIENT_PORT) + '/stop/', json={'jobs': related_old_jobs})
    logging.info(f"receive stop response with return={ret.text}")
    assert ret.status_code == 200


def progress(related_old_jobs, host):
    global available_jobs
    logging.info(f"send progress request with params: jobs={related_old_jobs}")
    ret = requests.post('http://' + host + ':' + str(CLIENT_PORT) + '/progress/', json={'jobs': related_old_jobs})
    logging.info(f"receive progress response with return={ret.text}")

    assert ret.status_code == 200
    for jid, prog in ret.json()['result'].items():
        available_jobs[jid]['progress'] += prog


def start(j, workers, host):
    global available_jobs
    params = {
        'job_id': j, 'job_type': available_jobs[j]['job_type'], 'batch_size': available_jobs[j]['batch_size'],
        'workers': workers,
    }
    logging.info(f"send start request with params: jobs={params}")
    ret = requests.post('http://' + host + ':' + str(CLIENT_PORT) + '/start/', json=params)
    assert ret.status_code == 200


def proceed_placement_for_host(host, host_worker_result, prev_host_worker_result):
    # map from job to worker for current round and for previous round respectively
    jobs_info, prev_jobs_info = {}, {}
    for w, jid in host_worker_result[host].items():
        if jid not in jobs_info.keys():
            jobs_info[jid] = []
        jobs_info[jid].append(w)

    if prev_host_worker_result.get(host):
        for w, jid in prev_host_worker_result[host].items():
            if jid not in prev_jobs_info.keys():
                prev_jobs_info[jid] = []
            prev_jobs_info[jid].append(w)

    for j, workers in jobs_info.items():
        if workers != prev_jobs_info.get(j):
            # at first, stop old jobs that occupy the workers used in the new job j
            related_old_jobs = []
            if prev_host_worker_result.get(host):
                for w in workers:
                    if prev_host_worker_result[host].get(w):
                        related_old_jobs.append(prev_host_worker_result[host][w])

            related_old_jobs = list(set(related_old_jobs))
            logging.info(f"for job {j}, {related_old_jobs} should be terminated.")
            stop(related_old_jobs, host)

    for j, workers in jobs_info.items():
        if workers != prev_jobs_info.get(j):
            start(j, workers, host)


def proceed_placement(host_worker_result, prev_host_worker_result):
    # proceed placement host by host
    thread_handlers = []
    for host in host_worker_result:
        thread = threading.Thread(target=proceed_placement_for_host, args=(host, host_worker_result, prev_host_worker_result))
        thread.start()
        thread_handlers.append(thread)

    for thread in thread_handlers:
        thread.join()

    logging.info(f't={t}, all placement threads complete!')


CLIENT_PORT = 37788
# length of scheduling window: 300 seconds
round_len = 60
prev_plan = {}
all_job_limit = 4
job_idx = 0
available_jobs = {}
job_stats = {}
gpus_each_host = 4
gpu_nums = {'a10': 4, 'v100': 4, 't4': 4, }
# gpu_types = ['v100', 'a10', 't4', ]
gpu_types = ['a10', ]
hosts_of_each_gpu_type = {
    'a10': ['172.22.47.92', ],
    't4': ['172.23.254.151', ],
    'v100': ['172.23.254.152', ],
}
prev_host_result = {}
prev_host_worker_result = {}
prev_jobs_location = {}
prev_jobs_worker_location = {}
t = 0
iteration_time = {
    # 'k80': {
    #     'vgg19': {
    #         512: {1: 0.815, 2: 0.555, 3: 0.466, 4: 0.434, },
    #         64: {1: 0.56, 2: 0.535, 3: 0.437, 4: 0.43, },
    #     },
    #     'resnet152': {
    #         512: {1: 2.17, 2: 1.23, 3: 0.917, 4: 0.73, },
    #         64: {1: 0.565, 2: 0.535, 3: 0.437, 4: 0.43, },
    #     },
    #
    #     'densenet121': {
    #         512: {1: 0.59, 2: 0.382, 3: 0.295, 4: 0.264, },
    #         64: {1: 0.296, 2: 0.265, 3: 0.26, 4: 0.26, },
    #     },
    # },
    't4': {
        'vgg19': {
            512: {1: 0.3224, 2: 0.222, 3: 0.235, 4: 0.225, },
            64: {1: 0.068, 2: 0.171, 3: 0.197, 4: 0.205, },
        },
        'resnet152': {
            512: {1: 0.4556, 2: 0.2462, 3: 0.225, 4: 0.1728, },
            64: {1: 0.119, 2: 0.123, 3: 0.13, 4: 0.133, },
        },
        'densenet121': {
            512: {1: 0.194, 2: 0.11, 3: 0.09, 4: 0.087, },
            64: {1: 0.075, 2: 0.075, 3: 0.077, 4: 0.078, },
        },
    },
    'a10': {
        'vgg19': {
            512: {1: 0.121, 2: 0.0819, 3: 0.085, 4: 0.1098, },
            64: {1: 0.027, 2: 0.056, 3: 0.071, 4: 0.105, },
        },
        'resnet152': {
            512: {1: 0.113, 2: 0.0795, 3: 0.0695, 4: 0.0765, },
            64: {1: 0.042, 2: 0.051, 3: 0.051, 4: 0.064, },
        },
        'densenet121': {
            512: {1: 0.067, 2: 0.0461, 3: 0.044, 4: 0.041, },
            64: {1: 0.041, 2: 0.040, 3: 0.038, 4: 0.039, },
        },
    },
    'v100': {
        'vgg19': {
            512: {1: 0.1, 2: 0.067, 3: 0.05, 4: 0.038, },
            64: {1: 0.021, 2: 0.039, 3: 0.03, 4: 0.022, },
        },
        'resnet152': {
            512: {1: 0.198, 2: 0.162, 3: 0.092, 4: 0.082, },
            64: {1: 0.067, 2: 0.067, 3: 0.065, 4: 0.065, },
        },
        'densenet121': {
            512: {1: 0.07, 2: 0.064, 3: 0.064, 4: 0.064, },
            64: {1: 0.06, 2: 0.058, 3: 0.058, 4: 0.058, },
        },
    },
}

if __name__ == '__main__':
    logging.basicConfig(
        filename='scheduler.log',
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    try:
        for t in range(100000):
            # update progress of jobs running on hosts according to the prev_plan
            update_progress_threads = []
            for j in prev_plan:
                thread = threading.Thread(target=progress([j, ], prev_jobs_location[j]))
                thread.start()
                update_progress_threads.append(thread)
            for thread in update_progress_threads:
                thread.join()

            # after updating progress, remove the completed jobs
            for j in available_jobs:
                if available_jobs[j]['progress'] >= available_jobs[j]['iteration']:
                    job_stats[j]['end_time'] = t
                    available_jobs.pop(j)

            # newly arrived jobs
            new_jobs = arrive(t)
            for new_job_id, new_job_attr in new_jobs.items():
                available_jobs[new_job_id] = copy.deepcopy(new_job_attr)
            for j in new_jobs:
                job_stats[j] = {}
                job_stats[j]['start_time'] = t

            # if no jobs, quit
            if not available_jobs:
                break

            plan = schedule()

            logging.info(f't={t}, schedule={plan}.')
            logging.info(f't={t}, available_jobs={available_jobs}.')

            allocate(plan, prev_plan)

            prev_plan = copy.deepcopy(plan)

            # sleep for round_len of time
            time.sleep(round_len)

            # break
        logging.info(job_stats)

        total_jct = 0
        for jid, j_attr in job_stats.items():
            total_jct += (j_attr['end_time'] - j_attr['start_time'])

        logging.info('total jct', total_jct)

    except Exception as e:
        logging.error(e)
