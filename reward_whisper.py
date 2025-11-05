import re
import math
from typing import List, Optional

import numpy as np
import torch

from xcodec2.modeling_xcodec2 import XCodec2Model

from transformers import pipeline, AutoProcessor
from transformers.models.whisper import WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from evaluate import load

wer = load("wer")
_asr_pipe = None
_codec_model = None
_text_normalizer = BasicTextNormalizer()
_asr_model = None
_asr_processor = None


def _lazy_load_codec_model() -> XCodec2Model:
    global _codec_model
    if _codec_model is None:
        _codec_model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")
        _codec_model.eval().cuda()
    return _codec_model


def _lazy_load_asr_pipe():
    global _asr_pipe
    if _asr_pipe is None:
        device = 0 if torch.cuda.is_available() else -1
        _asr_pipe = pipeline(
            task="automatic-speech-recognition",
            language="english",
            model="openai/whisper-large-v3",
            device=device,
        )
    return _asr_pipe


def _lazy_load_asr_nll():
    global _asr_model, _asr_processor
    if _asr_model is None or _asr_processor is None:
        model_id = "openai/whisper-large-v3"
        _asr_processor = AutoProcessor.from_pretrained(model_id)
        _asr_model = WhisperForConditionalGeneration.from_pretrained(model_id)
        _asr_model.eval()
        if torch.cuda.is_available():
            _asr_model.to("cuda")
    return _asr_model, _asr_processor

_TOKEN_PATTERN = re.compile(r"<\|s_(\d+)\|>")
_START_TAG = "<|SPEECH_GENERATION_START|>"
_END_TAG = "<|SPEECH_GENERATION_END|>"
_TU_START = "<|TEXT_UNDERSTANDING_START|>"
_TU_END = "<|TEXT_UNDERSTANDING_END|>"
_PROMPT_PREFIX = "Convert the text to speech:"


def _extract_ids_from_completion(completion: str) -> List[int]:
    return [int(m.group(1)) for m in _TOKEN_PATTERN.finditer(completion)]


def _slice_between_tags(text: str) -> str:
    if not text:
        return text
    start_idx = text.find(_START_TAG)
    if start_idx != -1:
        text = text[start_idx + len(_START_TAG):]
    end_idx = text.find(_END_TAG)
    if end_idx != -1:
        text = text[:end_idx]
    return text


def _decode_ids_to_waveform(ids: List[int], sampling_rate: int = 16000) -> Optional[np.ndarray]:
    if not ids:
        return None
    model = _lazy_load_codec_model()
    code = torch.tensor(ids, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    code = code.view(1, 1, -1)
    with torch.no_grad():
        gen_wav = model.decode_code(code)
    if not isinstance(gen_wav, torch.Tensor):
        return None
    wav = gen_wav[0, 0, :].detach().cpu().float().numpy()
    return wav

def _completion_to_text(comp) -> str:
    # Accepts either raw string or conversational list/dict and returns text content
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[0], dict) and "content" in comp[0]:
        return str(comp[0]["content"])  # single-turn assistant list
    if isinstance(comp, dict) and "content" in comp:
        return str(comp["content"])  # single message dict
    return str(comp)


def _extract_ref_from_prompt(prompt) -> Optional[str]:
    # Handle conversational prompt: list of {role, content}
    if isinstance(prompt, list):
        content = ""
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = str(msg.get("content", ""))
                break
    else:
        content = str(prompt)

    # Extract after the prefix
    idx = content.find(_PROMPT_PREFIX)
    if idx != -1:
        content = content[idx + len(_PROMPT_PREFIX):]

    # Remove optional TEXT_UNDERSTANDING tags
    tu_start = content.find(_TU_START)
    if tu_start != -1:
        content = content[tu_start + len(_TU_START):]
    tu_end = content.find(_TU_END)
    if tu_end != -1:
        content = content[:tu_end]

    content = content.strip()
    return content or None

def whisper_wer_reward(*, prompts, completions, text: Optional[List[str]] = None, **kwargs) -> List[Optional[float]]:
    """Compute reward from https://arxiv.org/pdf/2509.18798

    Expects the policy to emit audio code tokens like "<|s_123|><|s_456|>..." in `completions`.
    Uses XCodec2 to decode IDs to waveform, transcribes with Whisper, and computes WER.
    Returns a list of floats or None per sample.
    """
    asr = _lazy_load_asr_pipe()
    asr_model, asr_processor = _lazy_load_asr_nll()
    refs = [_extract_ref_from_prompt(p) for p in prompts]
    rewards: List[Optional[float]] = []

    alpha_wer = float(kwargs.get("alpha_wer", 3.0))
    alpha_nll = float(kwargs.get("alpha_nll", 3.0))

    lambda_wer = float(kwargs.get("lambda_wer", 0.6))
    lambda_nll = float(kwargs.get("lambda_nll", 0.4))

    eps = 1e-8
    for comp, ref_text in zip(completions, refs):

        comp_text = _completion_to_text(comp)
        comp_text = _slice_between_tags(comp_text)
        ids = _extract_ids_from_completion(comp_text)
        wav = _decode_ids_to_waveform(ids)

        # Normalize reference text before WER calculation
        ref_norm = _text_normalizer(ref_text) if ref_text is not None else ""
        asr_out = asr({"array": wav, "sampling_rate": 16000})
        hyp = asr_out["text"] if isinstance(asr_out, dict) else str(asr_out)
        hyp_norm = _text_normalizer(hyp)
        # Guard against empty reference (evaluate's WER divides by total words)
        ref_len = len(ref_norm.split())
        hyp_len = len(hyp_norm.split())
        if ref_len == 0:
            if hyp_len == 0:
                w = 0.0
            else:
                w = 1.0
        else:
            try:
                w = wer.compute(references=[ref_norm], predictions=[hyp_norm])
            except ZeroDivisionError:
                w = 1.0 if hyp_len > 0 else 0.0
        wer_reward = 1.0 - math.tanh(alpha_wer * float(w))

        # NLL component via Whisper model, mapped with exp(-loss)
        with torch.no_grad():
            proc = asr_processor(audio=wav, sampling_rate=16000, text=ref_norm, return_tensors="pt")
            if torch.cuda.is_available():
                for k, v in proc.items():
                    try:
                        proc[k] = v.to("cuda")
                    except Exception:
                        pass
            outputs = asr_model(**proc)
            loss = float(outputs.loss.detach().cpu().item())

        nll_reward = float(math.exp(-loss/alpha_nll))

        reward = (lambda_wer + lambda_nll) / ((lambda_wer/max(eps, wer_reward)) + (lambda_nll/max(eps, nll_reward)))
        reward = float(max(0.0, min(1.0, reward)))
        rewards.append(reward)
        print("ref_norm: ", ref_norm)
        print("hyp_norm: ", hyp_norm)
        print("wer_reward: ", wer_reward)
        print("nll_reward: ", nll_reward)
        print("reward: ", reward)


    return rewards



