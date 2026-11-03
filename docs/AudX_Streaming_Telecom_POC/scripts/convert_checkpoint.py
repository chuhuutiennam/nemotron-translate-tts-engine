"""Safely convert user-provided AudX full-model torch archive into weights-only.

Loads with `weights_only=True` + explicit verified class allowlist; does not
execute original model source or generic pickle code. Requires Torch >= 2.6.
"""
import argparse
from pathlib import Path
import sys
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audx_streaming import AudXCNNLSTM

parser=argparse.ArgumentParser()
parser.add_argument('--input',required=True)
parser.add_argument('--output',required=True)
a=parser.parse_args()
# Original class identity in pickle: models.cnn_lstm.CNNLSTM.
allowed=[(AudXCNNLSTM,'models.cnn_lstm.CNNLSTM'),
         nn.Sequential,nn.Conv2d,nn.ReLU,nn.LSTM,nn.Linear,set]
with torch.serialization.safe_globals(allowed):
    original = torch.load(a.input,map_location='cpu',weights_only=True)
if not isinstance(original,AudXCNNLSTM):
    raise TypeError('Unexpected checkpoint type')
if sum(p.numel() for p in original.parameters())!=9_282_753:
    raise RuntimeError('Unexpected architecture or checkpoint')
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
torch.save({'state_dict':original.state_dict(),
            'architecture':'AudXCNNLSTM_1_16_32_uniLSTM2_256_FC257',
            'sample_rate':16000,'n_fft':512,'hop_length':256},output)
print('Wrote',output,'bytes',output.stat().st_size)
