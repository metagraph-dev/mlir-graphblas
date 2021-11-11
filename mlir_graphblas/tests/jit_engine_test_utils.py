import numpy as np

MLIR_TYPE_TO_NP_TYPE = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    # 'f16': np.float16, # 16-bit floats don't seem to be supported in ctypes
    "f32": np.float32,
    "f64": np.float64,
}
