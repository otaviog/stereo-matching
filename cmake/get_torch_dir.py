from pathlib import Path
import torch

print((Path(torch.__file__).parent / 'share/cmake/Torch').absolute())
