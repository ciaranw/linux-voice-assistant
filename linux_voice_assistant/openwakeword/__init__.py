import json

from pathlib import Path
from typing import Union

from .tflite import TFLiteOpenWakeWord, TFLiteOpenWakeWordFeatures
from .onnx import OnnxOpenWakeWord, OnnxOpenWakeWordFeatures


def open_wake_word_from_config(
    config_path: Union[str, Path],
    libtensorflowlite_c_path: Union[str, Path],
) -> Union[TFLiteOpenWakeWord, OnnxOpenWakeWord]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    wake_word_model_path: Path = config_path.parent / config["model"]
    if wake_word_model_path.suffix == ".tflite":
        return TFLiteOpenWakeWord(
            id=Path(config["model"]).stem,
            wake_word=config["wake_word"],
            tflite_model=wake_word_model_path,
            libtensorflowlite_c_path=libtensorflowlite_c_path
        )
    
    elif wake_word_model_path.suffix == ".onnx":
        return OnnxOpenWakeWord(
            id=Path(config["model"]).stem,
            wake_word=config["wake_word"],
            onnx_model=wake_word_model_path,
        )
    else:
        raise ValueError(f"Unsupported open wake word model '{wake_word_model_path}'")
        