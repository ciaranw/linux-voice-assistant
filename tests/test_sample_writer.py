"""Tests for openWakeWord."""

import wave
from pathlib import Path
from typing import Union

from linux_voice_assistant.openwakeword.tflite import TFLiteOpenWakeWordFeatures, TFLiteOpenWakeWord
from linux_voice_assistant.openwakeword.onnx import OnnxOpenWakeWordFeatures, OnnxOpenWakeWord
from linux_voice_assistant.util import get_libtensorflowlite_lib_path
from linux_voice_assistant.sample_writer import SampleWriter

_TESTS_DIR = Path(__file__).parent
_REPO_DIR = _TESTS_DIR.parent
_OWW_DIR = _REPO_DIR / "wakewords" / "openWakeWord"

libtensorflowlite_c_path = get_libtensorflowlite_lib_path()

sample_writer = SampleWriter("127.0.0.1", 2055)

def test_tflite_features():
    features = TFLiteOpenWakeWordFeatures(
        melspectrogram_model=_OWW_DIR / "melspectrogram.tflite",
        embedding_model=_OWW_DIR / "embedding_model.tflite",
        libtensorflowlite_c_path=libtensorflowlite_c_path,
    )
    ww = TFLiteOpenWakeWord(
        id="hey_dick_head",
        wake_word="hey dick head",
        tflite_model=_OWW_DIR / "hey_dick_head.tflite",
        libtensorflowlite_c_path=libtensorflowlite_c_path,
    )

    process_test_features(features=features, ww=ww, sample_writer=sample_writer)

def test_onnx_features():
    features = OnnxOpenWakeWordFeatures(
        melspectrogram_model=_OWW_DIR / "melspectrogram.onnx",
        embedding_model=_OWW_DIR / "embedding_model.onnx",
    )
    ww = OnnxOpenWakeWord(
        id="hey_dick_head",
        wake_word="hey dick head",
        onnx_model=_OWW_DIR / "hey_dick_head.onnx",
    )

    process_test_features(features=features, ww=ww, sample_writer=sample_writer)


def process_test_features(
    features: Union[TFLiteOpenWakeWordFeatures, OnnxOpenWakeWordFeatures],
    ww: Union[TFLiteOpenWakeWord, OnnxOpenWakeWord],
    sample_writer: SampleWriter,
) -> None:
    max_prob = 0.0
    with wave.open(str(_TESTS_DIR / "hey_dick_head.wav"), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1

        for embeddings in features.process_streaming(
            wav_file.readframes(wav_file.getnframes())
        ):
            for prob in ww.process_streaming(embeddings):
                max_prob = max(max_prob, prob)

    sample_writer.write_sample_in_thread(features.get_audio_buffer())

    
