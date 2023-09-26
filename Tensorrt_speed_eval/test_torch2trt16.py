import torch
from torch2trt import torch2trt
from csfcn import get_pred_model
import time
import numpy as np
# create some regular pytorch model...
def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='fp16', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, ave batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))



model = get_pred_model(num_classes=19).eval().cuda()

# create example data
x = torch.ones((1, 3, 1024, 2048)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x], fp16_mode=True)
# torch.save(model_trt.state_dict(), 'sanet_trt.pth')
import time

benchmark(model_trt, input_shape=(1, 3, 1024, 2048), nruns=100)

