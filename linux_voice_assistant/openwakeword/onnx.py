"""openWakeWord onnx implementation."""

import ctypes as C
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Union

import numpy as np

c_void_p = C.c_void_p
c_int32 = C.c_int32
c_size_t = C.c_size_t

BATCH_SIZE: Final = 1

AUTOFILL_SECONDS: Final = 8
MAX_SECONDS: Final = 10

SAMPLE_RATE: Final = 16000  # 16Khz
_MAX_SAMPLES: Final = MAX_SECONDS * SAMPLE_RATE

SAMPLES_PER_CHUNK: Final = 1280  # 80 ms @ 16Khz
MS_PER_CHUNK: Final = SAMPLES_PER_CHUNK // SAMPLE_RATE

# window = 400, hop length = 160
MELS_PER_SECOND: Final = 97
MAX_MELS: Final = MAX_SECONDS * MELS_PER_SECOND
MEL_SAMPLES: Final = 1760
NUM_MELS: Final = 32

EMB_FEATURES: Final = 76  # 775 ms
EMB_STEP: Final = 8
MAX_EMB: Final = MAX_SECONDS * EMB_STEP
WW_FEATURES: Final = 96

MEL_SHAPE: Final = (BATCH_SIZE, MEL_SAMPLES)
EMB_SHAPE: Final = (BATCH_SIZE, EMB_FEATURES, NUM_MELS, 1)

# melspec = [batch x samples (min: 1280)] => [batch x 1 x window x mels (32)]
# stft window size: 25ms (400)
# stft window step: 10ms (160)
# mel band limits: 60Hz - 3800Hz
# mel frequency bins: 32
#
# embedding = [batch x window x mels (32) x 1] => [batch x 1 x 1 x features (96)]
# ww = [batch x window x features (96)] => [batch x probability]


class OnnxOpenWakeWord:
    def __init__(
        self,
        id: str,  # pylint: disable=redefined-builtin
        wake_word: str,
        onnx_model: Union[str, Path],
    ):
        try:
            import onnxruntime as ort

        except ImportError:
            raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")
        
        model_path = str(Path(onnx_model).resolve())
        
        if ".tflite" in model_path:
            raise ValueError("The onnx inference framework is selected, but tflite models were provided!")

        self.id = id
        self.wake_word = wake_word

        self.is_active = True

        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = 1
        sessionOptions.intra_op_num_threads = 1

        self.model = ort.InferenceSession(model_path, sess_options=sessionOptions,
                                                        providers=["CPUExecutionProvider"])

        # self.interpreter = self.lib.TfLiteInterpreterCreate(self.model, None)
        # self.lib.TfLiteInterpreterAllocateTensors(self.interpreter)

        self.model_inputs = self.model.get_inputs()[0].shape[1]
        self.model_outputs = self.model.get_outputs()[0].shape[1]

        # num_input_dims = self.lib.TfLiteTensorNumDims(self.input_tensor)
        # input_shape = [
        #     self.lib.TfLiteTensorDim(self.input_tensor, i)
        #     for i in range(num_input_dims)
        # ]
        self.input_windows = self.model_inputs

        self.new_embeddings: int = 0
        self.embeddings: np.ndarray = np.zeros(
            shape=(MAX_EMB, WW_FEATURES), dtype=np.float32
        )

    def process_streaming(self, embeddings: np.ndarray) -> Iterable[float]:
        """Generate probabilities from embeddings."""
        num_embedding_windows = embeddings.shape[2]

        # Shift
        self.embeddings[:-num_embedding_windows] = self.embeddings[
            num_embedding_windows:
        ]

        # Overwrite
        self.embeddings[-num_embedding_windows:] = embeddings[0, 0, :, :]
        self.new_embeddings = min(
            len(self.embeddings),
            self.new_embeddings + num_embedding_windows,
        )

        while self.new_embeddings >= self.input_windows:
            emb_tensor = np.zeros(
                shape=(1, self.input_windows, WW_FEATURES),
                dtype=np.float32,
            )

            emb_tensor[0, :] = self.embeddings[
                -self.new_embeddings : len(self.embeddings)
                - self.new_embeddings
                + self.input_windows
            ]
            self.new_embeddings = max(0, self.new_embeddings - 1)

            outputs = self.model.run(None, {self.model.get_inputs()[0].name: emb_tensor})

            yield outputs[0][0][0]

    @staticmethod
    def from_config(
        config_path: Union[str, Path],
    ) -> "OnnxOpenWakeWord":
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        return OnnxOpenWakeWord(
            id=Path(config["model"]).stem,
            wake_word=config["wake_word"],
            tflite_model=config_path.parent / config["model"],
        )


# -----------------------------------------------------------------------------


class OnnxOpenWakeWordFeatures:
    def __init__(
        self,
        melspectrogram_model: Union[str, Path],
        embedding_model: Union[str, Path],
        device = "cpu",
        ncpu = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")
        
        mel_path = str(Path(melspectrogram_model).resolve())
        emb_path = str(Path(embedding_model).resolve())
        
        if ".tflite" in mel_path or ".tflite" in emb_path:
            raise ValueError("The onnx inference framework is selected, but tflite models were provided!")
        
        # Initialize ONNX options
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = ncpu
        sessionOptions.intra_op_num_threads = ncpu

        provider = "CPUExecutionProvider"
        if device == "gpu":
            provider = "CUDAExecutionProvider"
        
        # Melspectrogram

        self.mel_model = ort.InferenceSession(mel_path, sess_options=sessionOptions,
                                                    providers=[provider])
        self.melspec_model_predict = lambda x: self.mel_model.run(None, {'input': x})

        # Audio embedding model
        self.emb_model = ort.InferenceSession(emb_path, sess_options=sessionOptions,
                                                    providers=[provider])
        self.embedding_model_predict = lambda x: self.emb_model.run(None, {'input_1': x})[0].squeeze()

        # State
        self.new_audio_samples: int = AUTOFILL_SECONDS * SAMPLE_RATE
        self.audio: np.ndarray = np.zeros(shape=(_MAX_SAMPLES,), dtype=np.float32)
        self.new_mels: int = 0
        self.mels: np.ndarray = np.zeros(shape=(MAX_MELS, NUM_MELS), dtype=np.float32)

    def process_streaming(self, audio_chunk: bytes) -> Iterable[np.ndarray]:
        """Generate embeddings from audio."""
        chunk_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)

        # Shift samples left
        self.audio[: -len(chunk_array)] = self.audio[len(chunk_array) :]

        # Add new samples to end
        self.audio[-len(chunk_array) :] = chunk_array
        self.new_audio_samples = min(
            len(self.audio),
            self.new_audio_samples + len(chunk_array),
        )

        while self.new_audio_samples >= MEL_SAMPLES:
            audio_tensor = np.zeros(shape=(BATCH_SIZE, MEL_SAMPLES), dtype=np.float32)
            audio_tensor[0, :] = self.audio[
                -self.new_audio_samples : len(self.audio)
                - self.new_audio_samples
                + MEL_SAMPLES
            ]
            audio_tensor = np.ascontiguousarray(audio_tensor)
            self.new_audio_samples = max(0, self.new_audio_samples - SAMPLES_PER_CHUNK)

            # Get melspectrogram
            outputs = self.melspec_model_predict(audio_tensor)
            mels = np.squeeze(outputs[0])

            

            mels = (mels / 10) + 2  # transform to fit embedding
            mels = mels.reshape((1, 1, -1, NUM_MELS))

            # Shift left
            num_mel_windows = mels.shape[2]
            self.mels[:-num_mel_windows] = self.mels[num_mel_windows:]

            # Overwrite
            self.mels[-num_mel_windows:] = mels[0, 0, :, :]
            self.new_mels = min(len(self.mels), self.new_mels + num_mel_windows)

            while self.new_mels >= EMB_FEATURES:
                mels_tensor = np.ascontiguousarray(
                    np.zeros(shape=EMB_SHAPE, dtype=np.float32)
                )
                mels_tensor[0, :, :, 0] = self.mels[
                    -self.new_mels : len(self.mels) - self.new_mels + EMB_FEATURES, :
                ]
                self.new_mels = max(0, self.new_mels - EMB_STEP)

                embedding = self.embedding_model_predict(mels_tensor)
                embedding = embedding.reshape((1, 1, -1, WW_FEATURES))
                yield embedding
