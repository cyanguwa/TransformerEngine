
# profiling
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --stats=true --output myprof python test_thd.py
nsys stats -r cuda_gpu_trace --format csv,column --output myprof myprof.sqlite


# parsing results
import pandas as pd
import torch
df=pd.read_csv('myprof_cuda_gpu_trace.csv')
t = df[df['Name'].str.contains("flash")]['Duration (ns)'].to_numpy()
tt = torch.Tensor(t).reshape(10, 2, 4).sum(dim=-1).mean(dim=0)/1e6 # tensor([0.4584, 0.4033])
print('thd vs sbhd speedup:', 1 - tt[1]/tt[0]) # 0.12020069808027922
