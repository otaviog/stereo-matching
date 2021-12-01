import torch

print(1 if torch.compiled_with_cxx11_abi() else 0)
