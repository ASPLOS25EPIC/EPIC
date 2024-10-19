import horovod.torch as hvd
# import paramiko
import os
import time
import torch


def find_duplicates(lst):
    seen = set()
    dups = set()
    for el in lst:
        if el in seen:
            dups.add(el)
        seen.add(el)
    return dups

# return communication mode: allreduce or allgather
def get_comm(params):
    comm_name = params.get('comm_mode', 'allreduce')
    return comm_name

def get_compressor(params):
    compress_name = params.get('compressor', 'none')
    compress_ratio = params.get('compress_ratio', 0.01)
    if compress_name == 'none':
        from compression.compressor.none import NoneCompressor
        compressor = NoneCompressor()
    elif compress_name == 'topk':
        from compression.compressor.topk import TopKCompressor
        compressor = TopKCompressor(compress_ratio,rank=hvd.rank())
    elif compress_name == 'fp16':
        from compression.compressor.fp16 import FP16Compressor
        compressor = FP16Compressor()
    elif compress_name == 'sign':
        from compression.compressor.sign import SignSGDCompressor
        compressor = SignSGDCompressor()
    else:
        raise NotImplementedError(compressor)

    return compressor

def get_memory(params):
    memory_name = params.get('memory', 'none')

    if memory_name == 'none':
        from compression.memory.none import NoneMemory
        memory = NoneMemory()

    elif memory_name == 'residual':
        from compression.memory.residual import ResidualMemory
        memory = ResidualMemory()  
    else:
        raise NotImplementedError(memory)
    return memory

def get_config(params):
    send_size_aresame = params.get('send_size_aresame', True)
    return send_size_aresame

def get_check(params):
    check = params.get('checkpoint', False)
    return check

# Special case:
# All dim==1 tensor should not be compressed
# ResNet: EF on the 'fc' will harm the performance of ADTOPK and AllchannelTopK
# VGG16: 'features.0' should not be compressed
# VGG16: EF on the 'classifier.6' will harm the performance
# LSTM: 'rnn.weight_hh' should not be compressed
def check_not_compress(params, name, tensor):
    
    if tensor.dim() == 1:
        return True
    if 'features.0' in name:
        return True
    if 'rnn.weight_hh' in name:
        return True

    return False


def check_not_ef(params, name, tensor):

    compressor_name = params.get('compressor', 'none')

    if 'adtopk' in compressor_name or 'alldimensiontopk' in compressor_name:
        if 'fc' in name:
            return True
    
    if 'classifier.6' in name:
        return True
    return False 

def get_pack(params):
    comm_name = params.get('pack', 1)
    return comm_name

def _to_cpu(data):
    if hasattr(data, 'cpu'):
        # Move tensor to CPU and return
        cpu_data = data.cpu()
        return cpu_data
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(v) for v in data)
    else:
        return data


# def load_ssh_config(hostname):
#     ssh_config = paramiko.SSHConfig()
#     config_path = os.path.expanduser("~/.ssh/config")
#     with open(config_path) as f:
#         ssh_config.parse(f)
#     host_config = ssh_config.lookup(hostname)
#     return host_config

# def set_ssh_connection(host_config):    
#     ssh = paramiko.SSHClient()
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
#     ssh.connect(
#         hostname=host_config['hostname'],
#         port=int(host_config.get('port', 22)),
#         username=host_config['user'],
#         key_filename=host_config.get('identityfile', None),
#         # You can add other parameters here as needed
#     )
    
#     sftp = ssh.open_sftp()
#     return sftp, ssh

# def close_ssh_connection(sftp, ssh):
#     sftp.close()
#     ssh.close()
    
# def transfer_checkpoint_to_remote(local_path, remote_path, sftp, ssh):
#     sftp.put(local_path, remote_path)


# class CheckpointThread(threading.Thread):
#     def __init__(self, diff_dict, checkpoint_path, save_event):
#         super(CheckpointThread, self).__init__()
#         self.diff_dict = diff_dict
#         self.checkpoint_path = checkpoint_path
#         self.save_event = save_event
#         self.daemon = True

#     def run(self):
#         while True:
#             self.save_event.wait()
#             # print("Saving checkpoint...")
#             torch.save(self.diff_dict, "/data/ycx/diff/test.pth.tar")
#             # print("Checkpoint saved.")
#             self.save_event.clear()

# try to use sub thread to save checkpoint but failed
# if hvd.rank() == 0:
#     self.save_event = threading.Event()
#     self.checkpoint_thread = CheckpointThread(self.diff_dict, "checkpoint.pth", self.save_event)
#     self.checkpoint_thread.start()

# self.save_event.set()

# def thread_save(diff ,filename):
#     begin = time.time()
#     torch.save(diff, filename)
#     end = time.time()
#     print("saving time: {}".format(end - begin))
#     return

# persist
# begin = time.time()
# diff = _to_cpu(self.diff_dict)
# end = time.time()
# print("to cpu time: {}".format(end - begin))

# process
# begin = time.time()
# self.ckpt_process = multiprocessing.Process(target=thread_save, args=(diff, filename))
# self.ckpt_process.start()
# end = time.time()
# print("process start time: {}".format(end - begin))

# thread
# begin = time.time()
# self.ckpt_thread = threading.Thread(target=thread_save, args=(diff, filename))
# self.ckpt_thread.start()
# end = time.time()
# print("thread start time: {}".format(end - begin))

# begin = time.time()
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     future = executor.submit(thread_save, diff, filename)
# end = time.time()
# print("concurrent.futures time: {}".format(end - begin))

#observation
# print("compress ckpt in {}s".format(sum(self.compress_time.values())))


# def transfer_checkpoint_to_remote(local_path, remote_path, hostname):
#     host_config = load_ssh_config(hostname)
    
#     ssh = paramiko.SSHClient()
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
#     ssh.connect(
#         hostname=host_config['hostname'],
#         port=int(host_config.get('port', 22)),
#         username=host_config['user'],
#         key_filename=host_config.get('identityfile', None),
#         # You can add other parameters here as needed
#     )
    
#     sftp = ssh.open_sftp()
#     sftp.put(local_path, remote_path)
#     sftp.close()
#     ssh.close()
#     print("Checkpoint transferred to remote machine.")

    
# def load_ssh_config(hostname):
#     ssh_config = paramiko.SSHConfig()
#     config_path = os.path.expanduser("~/.ssh/config")
#     with open(config_path) as f:
#         ssh_config.parse(f)
#     host_config = ssh_config.lookup(hostname)
#     return host_config
    


# import threading
# import concurrent.futures
# torch.multiprocessing.set_start_method('spawn', force=True)

# subprocess to save ckpt
# def save_process(queue,remote):
#     # for remote checkpoint save
#     if remote != '':
#         remote_queue = multiprocessing.Queue()
#         remote_process = multiprocessing.Process(target=save_remote_process, args=(remote_queue,remote, ))
#         remote_process.start()
#         print("remote subprocess start!")
    
#     while True:
#         data = queue.get()
#         if data is None:
#             if remote != '':
#                 remote_queue.put(None)
#             break
        
#         diff, filename = data
#         # print(diff["layer3.0.conv3.weight"]["tensors"][0].device)
#         # begin = time.time()
#         # diff = _to_cpu(diff)
#         # end = time.time()
#         # print("to cpu time: {}".format(end - begin))
#         begin = time.time()
#         torch.save(diff, filename)
#         end = time.time()
#         print("saved {} saving time: {}".format(filename, end - begin))
#         if remote != '':
#             remote_queue.put(filename)

# def save_remote_process(queue,remote):
#     host_config = load_ssh_config(remote)
#     sftp, ssh = set_ssh_connection(host_config)
#     while True:
#         name = queue.get()
#         if name is None:
#             close_ssh_connection(sftp, ssh)
#             break
#         begin = time.time()
#         new_path = name.replace("/diff/", "/remote/")
#         transfer_checkpoint_to_remote(name, new_path, sftp, ssh)
#         end = time.time()
#         print("remote {} saving time: {}".format(name, end - begin))

# def pack_save_differential_checkpoint(self, i, filename):
#     # data = _to_cpu(self.diff_dict)
#     data = self.diff_dict
#     self.diff_dict = {}
    
#     if self._comm_params.get('compressor', 'none') == 'topk':
        
#         self.batch_dict[i] = data
#         if i % self.pack == self.pack-1:
#             self.queue.put((self.batch_dict, filename))
#             self.batch_dict={}

#     elif self._comm_params.get('compressor', 'none') == 'fp16':
        
#         if i % self.pack == 0:
#             self.batch_dict = data
#         else:
#             for key in data.keys():
#                 self.batch_dict[key]['tensors'][0] += data[key]["tensors"][0]
        
#         if i % self.pack == self.pack-1:
#             self.queue.put((self.batch_dict, filename))
#             self.batch_dict={}  
#     else:
#         raise NotImplementedError("compressor not implemented")
