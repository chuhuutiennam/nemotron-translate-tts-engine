import unittest
import numpy as np
from audx_rtp_bridge import AudXRtpBridge
from audx_streaming import AudXCNNLSTM,offline_reference

class RtpBridgeTests(unittest.TestCase):
    def test_timestamp_alignment_and_no_duplicate_samples(self):
        model=AudXCNNLSTM().eval()
        a=AudXRtpBridge(model,device='cpu',rtp_clock_rate=8000)
        signal=np.random.default_rng(7).normal(0,0.03,size=8005).astype('float32')
        start=0xfffffff0
        out=[]
        emitted=0
        for pos in range(0,len(signal),320):
            b=a.push(signal[pos:pos+320],rtp_timestamp=(start+round(pos/2))&0xffffffff)
            if b is not None:
                self.assertEqual(b.rtp_timestamp,(start+round(emitted/2))&0xffffffff)
                out.append(b.float_pcm_16k)
                emitted+=len(b.float_pcm_16k)
        b=a.flush()
        if b is not None:
            self.assertEqual(b.rtp_timestamp,(start+round(emitted/2))&0xffffffff)
            out.append(b.float_pcm_16k)
        all_out=np.concatenate(out)
        self.assertEqual(len(all_out),len(signal))
        self.assertLess(float(np.max(np.abs(all_out-offline_reference(model,signal)))),1e-4)

if __name__=='__main__':unittest.main()
