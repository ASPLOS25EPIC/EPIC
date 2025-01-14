import torch
from compression import Compressor

class SignSGDCompressor(Compressor):

    def __init__(self):
        super().__init__(average=False)

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        tensor_compressed = tensor >= 0
        tensor_compressed = tensor_compressed.type(torch.uint8)
        return [tensor_compressed], shape

    def decompress(self, tensors, shape, name):
        """Decoding the signs to float format """
        sign_encode, _ = tensors
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = sum(tensors)
        sign = agged_tensor >= 0
        agged_tensor = sign * 2.0 - 1.0
        return agged_tensor