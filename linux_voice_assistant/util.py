"""Utility methods."""

import platform
import uuid
import os
from collections.abc import Callable
from typing import Optional
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_LIB_DIR = _MODULE_DIR.parent / "lib"

def get_mac() -> str:
    mac = uuid.getnode()
    mac_str = ":".join(f"{(mac >> i) & 0xff:02x}" for i in range(40, -1, -8))
    return mac_str


def call_all(*callables: Optional[Callable[[], None]]) -> None:
    for item in filter(None, callables):
        item()


def is_arm() -> bool:
    machine = platform.machine()
    return ("arm" in machine) or ("aarch" in machine)

def is_apple() -> bool:
    return "darwin" in platform.system()

def get_libtensorflowlite_lib_path() -> str:
    sysname = os.uname().sysname.lower()
    architecture = platform.machine()
    extension = 'dylib' if "darwin" == sysname else 'so'

    return _LIB_DIR / f"{sysname}_{architecture}" / f"libtensorflowlite_c.{extension}"
