from wyoming.client import AsyncTcpClient
from wyoming.audio import AudioStart, AudioChunk, AudioStop

import numpy as np
import threading
import asyncio
import logging

_LOGGER = logging.getLogger(__name__)

class SampleWriter(AsyncTcpClient):
  def __init__(self, host, port):
    super().__init__(host, port)

  def write_sample_in_thread(self, samples: np.ndarray):
    _thread = threading.Thread(target=asyncio.run, args=(self.write_sample(samples),))
    _thread.start()

  async def write_sample(self, samples: np.ndarray):
    _LOGGER.debug("sending sample to wyoming server")
    await self.connect()
    await self.write_event(AudioStart(16000, 2, 1).event())
    await self.write_event(AudioChunk(16000, 2, 1, samples).event())
    await self.write_event(AudioStop().event())
    await self.disconnect()
