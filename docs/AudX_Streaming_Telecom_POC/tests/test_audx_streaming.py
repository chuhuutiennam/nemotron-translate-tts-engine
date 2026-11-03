from __future__ import annotations
import os
from pathlib import Path
WEIGHTS = str(Path(__file__).resolve().parents[1] / "model" / "audx_weights_safe.pth")
import unittest
import numpy as np
import torch
from audx_streaming import AudXCNNLSTM, AudXStream, offline_reference, load_safe_weights, HOP

class AudXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = AudXCNNLSTM().eval()
        cls.rng = np.random.default_rng(783)

    def test_checkpoint_weights(self):
        path=WEIGHTS
        model=load_safe_weights(path)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9_282_753)
        self.assertFalse(model.lstm.bidirectional)
        self.assertEqual(model.lstm.num_layers, 2)

    def test_lstm_and_cnn_streaming_parity(self):
        rng = np.random.default_rng(567)
        feats = torch.from_numpy(rng.uniform(0,1,(1,85,257)).astype('float32'))
        with torch.inference_mode():
            ref = self.model(feats)[0].numpy()
            hidden=None
            predictions=[]
            base=0; next_frame=0
            for total in [1,3,4,8,12,19,31,49,64,80,85]:
                ready=total if total==85 else total-2
                if ready<=next_frame:continue
                local = feats[:,base:total,:]
                cnn = self.model.cnn(local.unsqueeze(1)).permute(0,2,1,3).contiguous().flatten(2)
                z,hidden=self.model.lstm(cnn[:,next_frame-base:ready-base,:],hidden)
                pred=self.model.fc(z)[0].numpy()
                predictions.append(pred)
                next_frame=ready;base=max(0,ready-2)
            online=np.concatenate(predictions)
        self.assertEqual(len(ref),len(online))
        self.assertLess(float(np.max(np.abs(ref-online))), 1e-4)

    def test_waveform_parity_irregular_chunks(self):
        rng = np.random.default_rng(201)
        for length in [131,510,1600,5056,16000,17777]:
            x = rng.normal(scale=0.05,size=length).astype('float32')
            expected=offline_reference(self.model,x)
            st=AudXStream(self.model)
            pieces=[];cursor=0
            for size in [319,511,1600,200,320,701,253,1200,77]*20:
                if cursor>=length:break
                take=x[cursor:cursor+size];cursor+=len(take)
                pieces.append(st.push(take))
            pieces.append(st.flush())
            actual=np.concatenate(pieces)
            self.assertEqual(len(x),len(actual))
            err=float(np.max(np.abs(actual-expected)))
            self.assertLess(err,0.00012,(length,err))
            self.assertEqual(st.received_samples,st.emitted_samples)

    def test_trained_checkpoint_streaming_parity(self):
        model=load_safe_weights(WEIGHTS)
        rng=np.random.default_rng(703)
        for packet in [320,1280,1600]:
            x=(0.07*rng.normal(size=16001)).astype('float32')
            reference=offline_reference(model,x)
            st=AudXStream(model)
            outputs=[]
            for off in range(0,len(x),packet):
                outputs.append(st.push(x[off:off+packet]))
            outputs.append(st.flush())
            streamed=np.concatenate(outputs)
            self.assertEqual(len(x),len(streamed))
            self.assertLess(float(np.max(np.abs(streamed-reference))),1e-5)

    def test_separate_stream_state(self):
        model=load_safe_weights(WEIGHTS)
        rng=np.random.default_rng(66)
        a=rng.normal(0,0.05,size=6155).astype('float32')
        b=rng.normal(0,0.06,size=7217).astype('float32')
        sa,sb=AudXStream(model),AudXStream(model)
        oa,ob=[],[]
        for offset in range(0,max(len(a),len(b)),640):
            if offset<len(a): oa.append(sa.push(a[offset:offset+640]))
            if offset<len(b): ob.append(sb.push(b[offset:offset+640]))
        oa.append(sa.flush());ob.append(sb.flush())
        self.assertLess(float(np.max(np.abs(np.concatenate(oa)-offline_reference(model,a)))),1e-5)
        self.assertLess(float(np.max(np.abs(np.concatenate(ob)-offline_reference(model,b)))),1e-5)

    def test_finite_and_closed(self):
        s=AudXStream(self.model)
        out=s.push(np.zeros(HOP*4,dtype='float32'))
        last=s.flush()
        self.assertEqual(len(out)+len(last),HOP*4)
        self.assertTrue(np.isfinite(np.r_[out,last]).all())
        with self.assertRaises(RuntimeError):s.push(np.zeros(HOP,dtype='float32'))
        with self.assertRaises(ValueError):AudXStream(self.model).push(np.array([np.nan],dtype='float32'))

if __name__=='__main__':unittest.main()

# Extra per-stream timestamp mapping test is in test_audx_rtp_bridge.py.
