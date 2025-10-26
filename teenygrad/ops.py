from enum import Enum, auto
from typing import Optional


class UnaryOps(Enum):
    NOOP = auto()
    EXP2 = auto()
    LOG2 = auto()
    CAST = auto()
    SIN = auto()
    SQRT = auto()
    RECIP = auto()
    NEG = auto()
    
    
class BinaryOps(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MAX = auto()
    MOD = auto()
    CMPLT = auto()
    
    
class ReduceOps(Enum):
    SUM = auto()
    MAX = auto()
    
    
class TernaryOps(Enum):
    MULACC = auto()
    WHERE = auto()
    
    
class MovementOps(Enum):
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()
    SHRINK = auto()
    STRIDE = auto()
    
    
class LoadOps(Enum):
    EMPTY = auto()
    RAND = auto()
    CONST = auto()
    FROM = auto()
    CONTIGUOUS = auto()
    CUSTOM = auto()
    
    
class Device:
    DEFAULT = "CPU"
    _buffers = ["CPU"]
    @staticmethod
    def canonicalize(device: Optional[str]) -> str:
        return "CPU"    
