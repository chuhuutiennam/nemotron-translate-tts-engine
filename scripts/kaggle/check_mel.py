"""Real mel check (docs/asr-feature-fix-design.md): the preprocessor of a .nemo, run by the NeMo of this venv, must
feed the model the same chunks as NeMo's reference chunking of the whole-utterance mel.

Prints the NeMo version, FilterbankFeatures.get_seq_len and the preprocessor config of the .nemo, refuses the config
the way the engine does when it loads the model, then compares 5 cases. Exit 1 when a case differs.

Usage: python scripts/kaggle/check_mel.py <model.nemo>   (venv with NeMo and torch; needs tests/test_runtime_features.py)
"""
import inspect
import io
import sys
import tarfile
from pathlib import Path

import nemo
import nemo.collections.asr as nemo_asr
import torch
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import test_runtime_features as t  # noqa: E402
from nemotron_testbed.runtime import _check_mel_config  # noqa: E402

if len(sys.argv) != 2:
    raise SystemExit("usage: check_mel.py <model.nemo>")

print("nemo", nemo.__version__)
print(inspect.getsource(FilterbankFeatures.get_seq_len))
cfg = None
with tarfile.open(sys.argv[1]) as tar:
    for member in tar:
        if member.name.endswith("model_config.yaml"):
            cfg = OmegaConf.load(io.StringIO(tar.extractfile(member).read().decode("utf-8")))
            break
if cfg is None:
    raise SystemExit(f"no model_config.yaml in {sys.argv[1]}")
pre_cfg = cfg.preprocessor
print(OmegaConf.to_yaml(pre_cfg))
_check_mel_config(pre_cfg)  # what the engine checks when it loads the model
with open_dict(pre_cfg):  # as runtime._load_model
    pre_cfg.dither = 0.0
    pre_cfg.pad_to = 0
    if pre_cfg.get("normalize", "NA") not in ("per_feature", "all_features"):
        pre_cfg.normalize = "None"
pre = nemo_asr.models.ASRModel.from_config_dict(pre_cfg).eval()
print("featurizer", type(pre.featurizer).__name__, "hop_length", pre.featurizer.hop_length, "n_fft", pre.featurizer.n_fft)


class RealPreprocessor:
    featurizer = pre.featurizer

    def __call__(self, input_signal, length):
        with torch.inference_mode():
            return pre(input_signal=input_signal, length=length)


t.FakePreprocessor = RealPreprocessor  # make_runtime() uses it
failed = 0
for steps, tail in ((20, 3000), (20, 0), (3, 1), (0, 5000), (0, 300)):
    runtime = t.make_runtime()
    runtime.mel_hop = int(pre.featurizer.hop_length)  # what _load_model reads
    runtime.mel_half_window = int(pre.featurizer.n_fft) // 2
    audio = t.utterance(steps * t.PIECE + tail)
    pieces = [audio[i * t.PIECE : (i + 1) * t.PIECE] for i in range(steps)]
    t.run(runtime, pieces, audio[steps * t.PIECE :])
    whole, length = RealPreprocessor()(torch.from_numpy(audio).unsqueeze(0), torch.tensor([audio.size]))
    ref = t.nemo_reference_chunks(whole[:, :, : int(length[0])])
    same = len(ref) == len(runtime.model.calls) and all(
        a[0].shape == b[0].shape and torch.allclose(a[0], b[0], atol=1e-4) and a[1:] == b[1:]
        for a, b in zip(runtime.model.calls, ref)
    )
    failed += not same
    print(f"{steps} STEP + {tail} samples: {len(runtime.model.calls)} chunks, same as NeMo reference: {same}")
sys.exit(1 if failed else 0)
