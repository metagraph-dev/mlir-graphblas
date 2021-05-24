#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "lowering.h"

using namespace std;
using namespace mlir;
using namespace mlir::sparse_tensor;

// make CSR tensor type
mlir::RankedTensorType getCSRTensorType(mlir::MLIRContext *context, mlir::FloatType valueType) {
    SmallVector<SparseTensorEncodingAttr::DimLevelType, 2> dlt;
    dlt.push_back(SparseTensorEncodingAttr::DimLevelType::Dense);
    dlt.push_back(SparseTensorEncodingAttr::DimLevelType::Compressed);
    unsigned ptr = 64;
    unsigned ind = 64;
    AffineMap map = {};

    RankedTensorType csrTensor = RankedTensorType::get(
        {-1, -1}, /* 2D, unknown size */
        valueType,
        SparseTensorEncodingAttr::get(context, dlt, map, ptr, ind));

    return csrTensor;
}
