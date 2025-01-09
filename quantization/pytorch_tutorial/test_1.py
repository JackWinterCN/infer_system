import torch
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
C, L = 3, 4
normal = torch.distributions.normal.Normal(0,1)
inputs = [normal.sample((C, L)), normal.sample((C, L))]
# print(inputs)

# >>>>>
r_max = -100000.0
r_min = 100000.0
for x in inputs:
    print(x)
    print(x.mean())
    print(x.min())
    print(x.max())
    r_min = min(r_min, x.min())
    r_max = max(r_max, x.max())
    r_interval = x.max() - x.min()
    print("r_interval = ", r_interval)
    s = r_interval/255.0
    print("s = ", s)
    z = 127 - x.max()/s
    print("z = ", z)

print("r_max = ", r_max, ", r_min = ", r_min)
r_interval_all = r_max - r_min
print("r_interval_all = ", r_interval_all)
s_all = r_interval_all/255.0
print("s_all = ", s_all, "1/s_all = ", 1.0/s_all)
z_all= 255 - r_max/s_all
print("z_all = ", z_all)

observers = [MinMaxObserver(), MovingAverageMinMaxObserver(), HistogramObserver()]
for obs in observers:
  for x in inputs:
    obs(x)
  print(obs.__class__.__name__, obs.calculate_qparams())


for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
  obs = MovingAverageMinMaxObserver(qscheme=qscheme)
  for x in inputs:
    obs(x)
  print(f"Qscheme: {qscheme} | {obs.calculate_qparams()}")

from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
obs = MovingAveragePerChannelMinMaxObserver(ch_axis=0)  # calculate qparams for all `C` channels separately
for x in inputs:
  obs(x)
print(obs.calculate_qparams())