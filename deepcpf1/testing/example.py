from deepcpf1 import deepcpf1
import numpy as np
sequences = np.array(['TGACTTTGAATGGAGTCGTGAGCGCAAGAACGCT', 
    'AAACTTTGAATGGAGTCGTGAGCGCAAGAACGCT','AAACTTTGAATGGAGTCGTGAGCGCAAGAACGAA'])
a=deepcpf1(sequences)

# {
#     "floatx": "float32",
#     "epsilon": 1e-07,
#     "backend": "tensorflow",
#     "image_data_format": "channels_last"
# }