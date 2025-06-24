from abc import ABC, abstractmethod

from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

class ModuleBase(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        assert torch.cuda.is_available(), "FlashCK needs CUDA."
        
