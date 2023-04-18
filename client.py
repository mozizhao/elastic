import logging
import os
import random
import signal
import subprocess
import threading
import flask
import psutil
from flask import Flask, request, jsonify
import json


PORT = 37788
app = Flask(__name__)
app.debug = True
get_port_lock = threading.Lock()
return_port_lock = threading.Lock()
worker_jid_association = {}
# jid_worker_association = {}
job_process_association = {}
jid_port_association = {}
used_port_pools = []
available_ports_low_bound = 40000
available_ports_upper_bound = 49999


def available_port():
    get_port_lock.acquire()
    p = random.choice(list(range(available_ports_low_bound, available_ports_upper_bound)))
    empty = False
    try:
        subprocess.check_output(f"lsof -i:{p}", shell=True).decode()
    except Exception as e:
        empty = True
    # randomly select a port that: 1. not used by other training jobs in this host 2. not used by other processes
    while (p in used_port_pools) or not empty:
        p = random.choice(list(range(available_ports_low_bound, available_ports_upper_bound)))

        try:
            subprocess.check_output(["lsof", f"-i:{p}"], shell=False).decode()
        except Exception as e:
            empty = True

    used_port_pools.append(p)
    get_port_lock.release()
    return p


def return_port(port):
    return_port_lock.acquire()
    used_port_pools.remove(port)
    return_port_lock.release()


def stop_job(jobs):
    if not jobs:
        return
    for j in jobs:
        if not job_process_association.get(j):
            logging.info(f"job {j} has already been terminated by other jobs.")
            continue
        pid = job_process_association[j]
        process = psutil.Process(pid)
        for proc in process.children(recursive=True):
            logging.info(f"killing subprocess {proc} of job {j} with pid={pid}")
            proc.kill()
        logging.info(f"killing main process of job {j} with pid={pid}")
        process.kill()
        # return the occupied port to the port pool.
        return_port(jid_port_association[j])
        # rm_progress(j)

        # delete association items
        job_process_association.pop(j)

        for w in [w for w in worker_jid_association if worker_jid_association[w] == j]:
            worker_jid_association.pop(w)


def collect_progress(jid):
    return float(subprocess.check_output(f"cat ./{jid}/progress.dat", shell=True).decode())


# def rm_progress(jid):
#     subprocess.call(f"rm -rf ./{jid}/progress.dat", shell=True)


def start_job(jid, job_type, batch_size, workers, init_progress):
    logging.info(f"receive start_job request with data {request.json}")
    worker_str = ""
    for w in workers:
        worker_str += f'{w},'
    worker_str = worker_str[0: -1]
    port = available_port()
    jid_port_association[jid] = port

    # p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={worker_str} python3 -m torch.distributed.launch --nproc_per_node {len(workers)} --master_port {port} ddp.py {job_type} {batch_size} {jid}", shell=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": f"{worker_str}",
    }
    p = subprocess.Popen(["python3", "-m", "torch.distributed.launch", "--nproc_per_node", f"{len(workers)}", "--master_port", f"{port}", "ddp.py", f"{job_type}", f"{batch_size}", f"{jid}", f"{init_progress}"], env=env, shell=False)
    logging.info(f"executed command: 'CUDA_VISIBLE_DEVICES={worker_str} python3 -m torch.distributed.launch --nproc_per_node {len(workers)} --master_port {port} ddp.py {job_type} {batch_size} {jid} {init_progress}'. Create training process [{p.pid}] for {jid}")

    # jid_worker_association[jid] = []
    for w in workers:
        # jid_worker_association[jid].append(w)
        worker_jid_association[w] = jid
    job_process_association[jid] = p.pid


@app.route('/stop/', methods=['post'])
def stop():
    logging.info(f"receive stop request with data {request.json}")
    old_jobs = request.json['jobs']
    # jid, workers = json.loads(param).keys()[0], json.loads(param).values()[0]
    stop_job(old_jobs)
    return 'success'


@app.route('/progress/', methods=['post'])
def progress():
    jobs = request.json['jobs']
    if not jobs:
        return jsonify({'result': {}})
    else:
        result = {}
        for jid in jobs:
            progress = collect_progress(jid)
            result[jid] = progress

        return jsonify({'result': result})


@app.route('/start/', methods=['post'])
def start():
    jid, job_type, batch_size, workers, init_progress = request.json['job_id'], request.json['job_type'], request.json['batch_size'], request.json['workers'], request.json['progress']

    start_job(jid, job_type, batch_size, workers, init_progress)
    return 'success'


if __name__ == '__main__':
    logging.basicConfig(
        filename='client.log',
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    app.run(host='0.0.0.0', port=PORT)
