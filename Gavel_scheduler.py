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


def throughput(gpu_type, job_type, batch_size, worker):
    return 1 / cal_iteration_time(gpu_type, job_type, batch_size, worker)


def schedule():
    # list to store executable jobs of each GPU type in this round
    temp_queues = []
    result = {}
    for i in range(len(queues)):
        temp_queues.append([])

        gpu_type = gpu_types[i]
        available_workers = gpu_nums[gpu_type]
        # sorted by the processing time (as we mimic SJT rather than SRPT) in this gpu type
        sorted_jobs = sorted(
            queues[i],
            key=lambda j: (available_jobs[j]['iteration'] - available_jobs[j]['progress']) / throughput(gpu_type, available_jobs[j]['job_type'], available_jobs[j]['batch_size'], available_jobs[j]['demand']),
            reverse=False
        )
        # allocate GPUs to jobs by processing time in ascending order
        for j in sorted_jobs:
            if available_workers >= available_jobs[j]['demand']:
                available_workers -= available_jobs[j]['demand']
                temp_queues[i].append(j)

    for i in range(len(temp_queues)):
        gpu_type = gpu_types[i]
        for j in temp_queues[i]:
            result[j] = {'type': gpu_type, 'num': available_jobs[j]['demand'], 'job_type': available_jobs[j]['job_type'], 'bs': available_jobs[j]['batch_size']}

    return result


# calculates the iteration time of iteration time or the fitted iteration time, depending on the user's choice.
def cal_iteration_time(gpu_type, job_type, batch_size, gpu_count):
    return iteration_time[gpu_type][job_type][batch_size][gpu_count] / round_len


def allocate(plan, prev_plan):
    # the first two record the data where hosts are keys, while the last two record the data where jobs are keys
    global t, prev_host_result, prev_host_worker_result, prev_jobs_location
    # record host information and worker information respectively in this round
    host_result, host_worker_result = {}, {}

    # variables initialization
    for gpu_type in gpu_types:
        for host in hosts_of_each_gpu_type[gpu_type]:
            # used_host_worker_nums[host] = 0
            host_result[host] = []
            host_worker_result[host] = {}

    # process jobs type by type
    for gpu_type in gpu_types:
        host = hosts_of_each_gpu_type[gpu_type][0]
        jobs = list(filter((lambda j: plan[j]['type'] == gpu_type), plan.keys()))

        count = 0
        for j in jobs:
            if j not in host_result[host]:
                host_result[host].append(j)
            for _ in range(plan[j]['num']):
                host_worker_result[host][count] = j
                count += 1

    # start control plane
    logging.info(f"t={t}, placement host_result={host_result}, host_worker_result={host_worker_result}")
    proceed_placement(host_result, prev_host_result, host_worker_result, prev_host_worker_result)

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
        available_jobs[jid]['progress'] = prog


def start(j, workers, host):
    global available_jobs
    params = {
        'host': host, 'job_id': j, 'job_type': available_jobs[j]['job_type'],
        'batch_size': available_jobs[j]['batch_size'], 'progress': available_jobs[j]['progress'], 'workers': workers,
    }
    logging.info(f"send start request with params: params={params}")
    ret = requests.post('http://' + host + ':' + str(CLIENT_PORT) + '/start/', json=params)
    assert ret.status_code == 200


def proceed_placement_for_host(host, host_result, prev_host_result, host_worker_result, prev_host_worker_result):

    related_old_jobs = []
    if prev_host_result.get(host):
        related_old_jobs = copy.deepcopy(prev_host_result[host])

    if related_old_jobs:
        stop(related_old_jobs, host)

    executable_jobs = copy.deepcopy(host_result[host])
    jobs_info = {}
    for j in executable_jobs:
        if j not in jobs_info:
            workers = list(filter((lambda w: host_worker_result[host][w] == j), host_worker_result[host]))
            jobs_info[j] = workers

    for j, workers in jobs_info.items():
        start(j, workers, host)


def proceed_placement(host_result, prev_host_result, host_worker_result, prev_host_worker_result):
    # proceed placement host by host
    thread_handlers = []
    for host in host_worker_result:
        thread = threading.Thread(target=proceed_placement_for_host, args=(host, host_result, prev_host_result, host_worker_result, prev_host_worker_result))
        thread.start()
        thread_handlers.append(thread)

    for thread in thread_handlers:
        thread.join()

    logging.info(f't={t}, all placement threads complete!')



CLIENT_PORT = 37788
# length of scheduling window: 300 seconds
round_len = 120
prev_plan = {}
all_job_limit = 4
job_idx = 0
available_jobs = {}
job_stats = {}
gpus_each_host = 4
gpu_nums = {'a10': 4, 'v100': 4, 't4': 4, }
# gpu_types = ['v100', 'a10', 't4', ]
gpu_types = ['a10', 't4', 'v100', ]
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
# queues for accommodating jobs to different GPU types
queues = [[], [], []]
batch_time = {
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
            512: {1: 0.315, 2: 0.215, 3: 0.238, 4: 0.231, },
            64: {1: 0.072, 2: 0.182, 3: 0.2, 4: 0.214, },
        },
        'resnet152': {
            512: {1: 0.365, 2: 0.222, 3: 0.185, 4: 0.171, },
            64: {1: 0.119, 2: 0.133, 3: 0.14, 4: 0.144, },
        },
        'densenet121': {
            512: {1: 0.167, 2: 0.12, 3: 0.1, 4: 0.1, },
            64: {1: 0.097, 2: 0.093, 3: 0.09, 4: 0.086, },
        },
    },
    'a10': {
        'vgg19': {
            512: {1: 0.123, 2: 0.082, 3: 0.085, 4: 0.088, },
            64: {1: 0.0318, 2: 0.059, 3: 0.0689, 4: 0.0719, },
        },
        'resnet152': {
            512: {1: 0.118, 2: 0.091, 3: 0.082, 4: 0.079, },
            64: {1: 0.07, 2: 0.072, 3: 0.0672, 4: 0.065, },
        },
        'densenet121': {
            512: {1: 0.075, 2: 0.063, 3: 0.06, 4: 0.063, },
            64: {1: 0.06, 2: 0.0575, 3: 0.0584, 4: 0.0555, },
        },
    },
    'v100': {
        'vgg19': {
            512: {1: 0.105, 2: 0.064, 3: 0.05, 4: 0.038, },
            64: {1: 0.024, 2: 0.044, 3: 0.031, 4: 0.023, },
        },
        'resnet152': {
            512: {1: 0.2, 2: 0.165, 3: 0.096, 4: 0.087, },
            64: {1: 0.076, 2: 0.075, 3: 0.074, 4: 0.072, },
        },
        'densenet121': {
            512: {1: 0.082, 2: 0.068, 3: 0.068, 4: 0.068, },
            64: {1: 0.07, 2: 0.066, 3: 0.065, 4: 0.068, },
        },
    },
}

iteration_time = {
    't4': {
        'vgg19': {
            512: {1: 37.4, 2: 25, 3: 26, 4: 24.5, },
            64: {1: 63.6, 2: 149, 3: 159, 4: 172, },
        },
        'resnet152': {
            512: {1: 43.5, 2: 25.2, 3: 20, 4: 18.2, },
            64: {1: 89.2, 2: 103.4, 3: 104.2, 4: 111.8, },
        },
        'densenet121': {
            512: {1: 22.3, 2: 14.5, 3: 11.5, 4: 13.6, },
            64: {1: 77.8, 2: 70, 3: 68.2, 4: 66.4, },
        },
    },
    'a10': {
        'vgg19': {
            512: {1: 16.1, 2: 10.2, 3: 9.7, 4: 9.1, },
            64: {1: 29.8, 2: 47.5, 3: 51., 4: 55, },
        },
        'resnet152': {
            512: {1: 15.4, 2: 10.6, 3: 9.2, 4: 8.5, },
            64: {1: 58.1, 2: 53.4, 3: 48.7, 4: 53, },
        },
        'densenet121': {
            512: {1: 11, 2: 7.8, 3: 7.2, 4: 6.8, },
            64: {1: 50.4, 2: 46., 3: 43.4, 4: 44., },
        },
    },
    'v100': {
        'vgg19': {
            512: {1: 16.3, 2: 9.9, 3: 7.3, 4: 5.6, },
            64: {1: 23.1, 2: 25., 3: 25.5, 4: 20, },
        },
        'resnet152': {
            512: {1: 27.1, 2: 20.6, 3: 13., 4: 11.5, },
            64: {1: 61.3, 2: 63., 3: 60., 4: 59., },
        },
        'densenet121': {
            512: {1: 13.6, 2: 13., 3: 11.7, 4: 10.9, },
            64: {1: 56.1, 2: 72., 3: 68.5, 4: 64., },
        },
    },
}


if __name__ == '__main__':
    logging.basicConfig(
        filename='Gavel.log',
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
            for j in new_jobs:
                # random.seed(SEED_BASE + job_idx)
                random.choice(queues).append(j)
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
        logging.exception(e)
