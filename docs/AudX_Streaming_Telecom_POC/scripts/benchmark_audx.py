"""No-I/O local smoke benchmark. Hardware/production results require target machine."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from audx_streaming import AudXStream,offline_reference,load_safe_weights

parser=argparse.ArgumentParser()
parser.add_argument('--weights',default=str(Path(__file__).resolve().parents[1] / 'model' / 'audx_weights_safe.pth'))
parser.add_argument('--device',default='cpu')
parser.add_argument('--threads',type=int,default=2)
parser.add_argument('--seconds',type=float,default=4.0)
parser.add_argument('--packet-ms',type=float,default=80.0)
args=parser.parse_args()
torch.set_num_threads(args.threads)
model=load_safe_weights(args.weights,args.device)
sample_rate=16000
n=int(args.seconds*sample_rate)
t=np.arange(n,dtype=np.float64)/sample_rate
wave=(0.04*np.sin(2*np.pi*220*t)+0.02*np.sin(2*np.pi*410*t)+0.008*np.random.default_rng(5).standard_normal(n)).astype('float32')
# warmup is excluded
_ = offline_reference(model,wave[:min(16000,n)],args.device)
stream=AudXStream(model,args.device)
block=max(1,int(sample_rate*args.packet_ms/1000))
latencies=[]; pieces=[]; first=None
start=time.perf_counter()
for off in range(0,n,block):
  s=time.perf_counter()
  y=stream.push(wave[off:off+block]);elapsed=(time.perf_counter()-s)*1000
  latencies.append(elapsed)
  pieces.append(y)
  if first is None and len(y):first=(off+len(wave[off:off+block]))/sample_rate
pieces.append(stream.flush())
total=time.perf_counter()-start
result=np.concatenate(pieces)
ref=offline_reference(model,wave,args.device)
print(f'weights={args.weights} device={args.device} threads={args.threads}')
print('waveform_s=',len(wave)/sample_rate,'packet_ms=',args.packet_ms)
print('output_samples=',len(result),'input_samples=',len(wave))
print('max_abs_error_offline=',float(np.max(np.abs(ref-result))))
print('first_output_after_audio_s=',first)
print('p50_push_ms=',round(float(np.percentile(latencies,50)),2),'p95=',round(float(np.percentile(latencies,95)),2),'max=',round(float(max(latencies)),2))
print('compute_RTF=',round(total/args.seconds,4),'total_wall_s=',round(total,4))
print('model_inference_ms=',round(stream.timings.model_ms,1),'dsp_ms=',round(stream.timings.dsp_ms,1),'model_calls=',stream.timings.calls)
