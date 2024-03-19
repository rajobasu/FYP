from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def print_gpu_utilization(id):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(id)  # using cuda:1
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024 ** 2
