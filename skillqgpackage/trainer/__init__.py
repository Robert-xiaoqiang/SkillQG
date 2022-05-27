__all__ = [
    'CLMTrainerModule',
    'Seq2SeqLMTrainerModule',
    'RLCLMTrainerModule'
]

from .CLMTrainerModule import CLMTrainerModule
from .Seq2SeqLMTrainerModule import Seq2SeqLMTrainerModule
from .RLCLMTrainerModule import RLCLMTrainerModule

'''
These pre-importing classes will overwrite/hide their corresponding parent-level modules (with the same names).
i.e., `class CLMTrainerModule/Seq2SeqLMTrainerModule/RLCLMTrainerModule` hides `module CLMTrainerModule/Seq2SeqLMTrainerModule/RLCLMTrainerModule`.
Therefore, when using the wildcard-based subset import (i.e., `from trainer import *`), we will acquire a total of three identifiers, including `class CLMTrainerModule`, `class Seq2SeqLMTrainerModule`, and `class RLCLMTrainerModule`.
Whatever the order of pre-importing and wildcard-subset-importing statements is, we can assume that python interpreter will parse the subset-importing firstly, and then resolve the pre-importing things with possible overwriting/hiding.
'''