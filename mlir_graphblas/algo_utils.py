import math
from .mlir_builder import MLIRVar


# def ensure_constant(irb, x, typestr):
#     if isinstance(x, MLIRVar):
#         return x
#     ret = irb.new_var(typestr)
#     irb.add_statement(f"{ret.assign} = constant {x} : {ret.type}")
#     return ret


def euclidean_distance(irb, x, y):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances
    # Assumes x and y are dense tensors
    x2 = irb.graphblas.intersect(x, x, "times")
    xx = irb.graphblas.reduce_to_vector(x2, "plus", axis=1)
    y2 = irb.graphblas.intersect(y, y, "times")
    yy = irb.graphblas.reduce_to_vector(y2, "plus", axis=1)
    yT = irb.graphblas.transpose(y, "tensor<?x?xf64, #CSC64>")
    yT = irb.graphblas.apply(yT, "times", right=irb.arith.constant(-2, "f64"))
    rv = irb.graphblas.matrix_multiply(x, yT, "plus_times")
    rv = irb.graphblas.matrix_multiply(
        irb.graphblas.diag(xx, "tensor<?x?xf64, #CSR64>"), rv, "min_plus"
    )
    rv = irb.graphblas.matrix_multiply(
        rv, irb.graphblas.diag(yy, "tensor<?x?xf64, #CSC64>"), "min_plus"
    )
    rv = irb.graphblas.apply(rv, "sqrt")
    return rv


def haversine_distance(
    irb, many_lat, many_lon, single_lat, single_lon, *, radius=6371.0, to_radians=True
):
    """Compute the distances between many_{lat,lon} and single_{lat,lon}"""
    # many_lat (and many_lon) may be a Matrix or a Vector
    # single_lat (and single_lon) must be a Vector
    cf05 = irb.arith.constant(0.5, "f64")
    cf2 = irb.arith.constant(2.0, "f64")
    cf_2radius = irb.arith.constant(2 * radius, "f64")
    if to_radians:
        cf_rad = irb.arith.constant(math.tau / 360, "f64")
        many_lat = irb.graphblas.apply(many_lat, "times", right=cf_rad)
        many_lon = irb.graphblas.apply(many_lon, "times", right=cf_rad)
        single_lat = irb.graphblas.apply(single_lat, "times", right=cf_rad)
        single_lon = irb.graphblas.apply(single_lon, "times", right=cf_rad)
    if many_lat.type.encoding.rank == 1:  # Vector
        diff_lat = irb.graphblas.intersect(single_lat, many_lat, "minus")
        diff_lon = irb.graphblas.intersect(single_lon, many_lon, "minus")
        cos_terms = irb.graphblas.intersect(
            irb.graphblas.apply(single_lat, "cos"),
            irb.graphblas.apply(many_lat, "cos"),
            "times",
        )
    else:  # Matrix
        single_lat = irb.graphblas.diag(single_lat, "tensor<?x?xf64, #CSR64>")
        single_lon = irb.graphblas.diag(single_lon, "tensor<?x?xf64, #CSR64>")
        diff_lat = irb.graphblas.matrix_multiply(single_lat, many_lat, "any_minus")
        diff_lon = irb.graphblas.matrix_multiply(single_lon, many_lon, "any_minus")
        cos_terms = irb.graphblas.matrix_multiply(
            irb.graphblas.apply(single_lat, "cos"),
            irb.graphblas.apply(many_lat, "cos"),
            "any_times",
        )
    a = irb.graphblas.intersect(
        irb.graphblas.apply(
            irb.graphblas.apply(
                irb.graphblas.apply(diff_lat, "times", left=cf05), "sin"
            ),
            "pow",
            right=cf2,
        ),
        irb.graphblas.intersect(
            cos_terms,
            irb.graphblas.apply(
                irb.graphblas.apply(
                    irb.graphblas.apply(diff_lon, "times", left=cf05), "sin"
                ),
                "pow",
                right=cf2,
            ),
            "times",
        ),
        "plus",
    )
    return irb.graphblas.apply(
        irb.graphblas.apply(irb.graphblas.apply(a, "sqrt"), "asin"),
        "times",
        left=cf_2radius,
    )


def normprob(irb, x):
    x_max = irb.graphblas.reduce_to_vector(x, "max", axis=0)
    x = irb.graphblas.matrix_multiply(
        x, irb.graphblas.diag(x_max, "tensor<?x?xf64, #CSC64>"), "min_minus"
    )
    x_exp = irb.graphblas.apply(x, "exp")
    x_exp_sum = irb.graphblas.reduce_to_vector(x_exp, "plus", axis=0)
    rv = irb.graphblas.matrix_multiply(
        x_exp, irb.graphblas.diag(x_exp_sum, "tensor<?x?xf64, #CSC64>"), "min_div"
    )
    rv = irb.graphblas.apply(rv, "log")
    return rv


def tensor_arange(irb, size, dtype="index"):
    # %ret = tensor.generate %size {
    # ^bb0(%i: index):
    #   tensor.yield %i : index
    # } : tensor<?xindex>
    ret = irb.new_var(f"tensor<?x{dtype}>")
    size = irb.arith.constant(size, "index")
    irb.add_statement(f"{ret.assign} = tensor.generate {size} {{")
    irb.add_statement(f"^bb0(%i: {dtype}):")
    irb.add_statement(f"  tensor.yield %i : {dtype}")
    irb.add_statement(f"}} : {ret.type}")
    return ret


def tensor_extract_col(irb, tensor, n):
    n = irb.arith.constant(n, "index")
    num_rows = irb.tensor.dim(tensor, 0)
    ret = irb.new_var(f"tensor<?x1x{tensor.type.value_type}>")
    irb.add_statement(
        f"{ret.assign} = tensor.extract_slice {tensor}[0, {n}][{num_rows}, 1][1, 1] : {tensor.type} to {ret.type}"
    )
    return ret


def tensor_extract_row(irb, tensor, n):
    n = irb.arith.constant(n, "index")
    num_cols = irb.tensor.dim(tensor, 1)
    ret = irb.new_var(f"tensor<1x?x{tensor.type.value_type}>")
    irb.add_statement(
        f"{ret.assign} = tensor.extract_slice {tensor}[{n}, 0][1, {num_cols}][1, 1] : {tensor.type} to {ret.type}"
    )
    return ret


def tensor_insert_col(irb, slice, tensor, n):
    n = irb.arith.constant(n, "index")
    num_rows = irb.tensor.dim(tensor, 0)
    ret = irb.new_var(tensor.type)
    irb.add_statement(
        f"{ret.assign} = tensor.insert_slice {slice} into {tensor}[0, {n}][{num_rows}, 1][1, 1] : {slice.type} into {tensor.type}"
    )
    return ret


def tensor_insert_row(irb, slice, tensor, n):
    n = irb.arith.constant(n, "index")
    num_cols = irb.tensor.dim(tensor, 1)
    ret = irb.new_var(tensor.type)
    irb.add_statement(
        f"{ret.assign} = tensor.insert_slice {slice} into {tensor}[{n}, 0][1, {num_cols}][1, 1] : {slice.type} into {tensor.type}"
    )
    return ret


def tensor_fill(irb, size, value, dtype):
    # %ret = tensor.generate %size {
    # ^bb0(%i: index):
    #   tensor.yield %value : i64
    # } : tensor<?xi64>
    ret = irb.new_var(f"tensor<?x{dtype}>")
    size = irb.arith.constant(size, "index")
    value = irb.arith.constant(value, dtype)
    irb.add_statement(f"{ret.assign} = tensor.generate {size} {{")
    irb.add_statement("^bb0(%i: index):")
    irb.add_statement(f"  tensor.yield {value} : {value.type}")
    irb.add_statement(f"}} : {ret.type}")
    return ret


def tensor_to_col(irb, tensor):
    ret = irb.new_var(f"tensor<?x1x{tensor.type.value_type}>")
    irb.add_statement(
        f"{ret.assign} = tensor.expand_shape {tensor} [[0, 1]] : {tensor.type} into {ret.type}"
    )
    return ret


def tensor_to_row(irb, tensor):
    ret = irb.new_var(f"tensor<1x?x{tensor.type.value_type}>")
    irb.add_statement(
        f"{ret.assign} = tensor.expand_shape {tensor} [[0, 1]] : {tensor.type} into {ret.type}"
    )
    return ret


def tensor_to_1d(irb, tensor):
    # Works if tensor is either a single row or single column 2D tensor
    ret = irb.new_var(f"tensor<?x{tensor.type.value_type}>")
    irb.add_statement(
        f"{ret.assign} = tensor.collapse_shape {tensor} [[0, 1]] : {tensor.type} into {ret.type}"
    )
    return ret
