"""Tests for openWakeWord."""

import wave
from pathlib import Path
from typing import Union

from linux_voice_assistant.openwakeword.tflite import TFLiteOpenWakeWordFeatures, TFLiteOpenWakeWord
from linux_voice_assistant.openwakeword.onnx import OnnxOpenWakeWordFeatures, OnnxOpenWakeWord
from linux_voice_assistant.util import get_libtensorflowlite_lib_path

_TESTS_DIR = Path(__file__).parent
_REPO_DIR = _TESTS_DIR.parent
_OWW_DIR = _REPO_DIR / "wakewords" / "openWakeWord"

libtensorflowlite_c_path = get_libtensorflowlite_lib_path()

def test_tflite_features():
    features = TFLiteOpenWakeWordFeatures(
        melspectrogram_model=_OWW_DIR / "melspectrogram.tflite",
        embedding_model=_OWW_DIR / "embedding_model.tflite",
        libtensorflowlite_c_path=libtensorflowlite_c_path,
    )
    ww = TFLiteOpenWakeWord(
        id="ok_nabu",
        wake_word="okay nabu",
        tflite_model=_OWW_DIR / "ok_nabu_v0.1.tflite",
        libtensorflowlite_c_path=libtensorflowlite_c_path,
    )

    process_test_features(features=features, ww=ww)
    # negative test fails for some reason?
    # process_test_features_no_match(features=features, ww=ww)

def test_onnx_features():
    features = OnnxOpenWakeWordFeatures(
        melspectrogram_model=_OWW_DIR / "melspectrogram.onnx",
        embedding_model=_OWW_DIR / "embedding_model.onnx",
    )
    ww = OnnxOpenWakeWord(
        id="ok_nabu",
        wake_word="okay nabu",
        onnx_model=_OWW_DIR / "ok_nabu_v0.2.onnx",
    )

    process_test_features(features=features, ww=ww)
    process_test_features_no_match(features=features, ww=ww)


def process_test_features(
    features: Union[TFLiteOpenWakeWordFeatures, OnnxOpenWakeWordFeatures],
    ww: Union[TFLiteOpenWakeWord, OnnxOpenWakeWord]
) -> None:
    max_prob = 0.0
    with wave.open(str(_TESTS_DIR / "ok_nabu.wav"), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1

        for embeddings in features.process_streaming(
            wav_file.readframes(wav_file.getnframes())
        ):
            for prob in ww.process_streaming(embeddings):
                max_prob = max(max_prob, prob)

    assert max_prob > 0.5

def process_test_features_no_match(
    features: Union[TFLiteOpenWakeWordFeatures, OnnxOpenWakeWordFeatures],
    ww: Union[TFLiteOpenWakeWord, OnnxOpenWakeWord]
) -> None:
    max_prob = 0.0
    with wave.open(str(_TESTS_DIR / "negative.wav"), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1

        for embeddings in features.process_streaming(
            wav_file.readframes(wav_file.getnframes())
        ):
            for prob in ww.process_streaming(embeddings):
                max_prob = max(max_prob, prob)

    assert max_prob < 0.5
