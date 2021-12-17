Supported GraphBLAS Spec Operations
===================================

.. csv-table:: Supported GraphBLAS Spec operations
    :header: Operation, Matrix, Vector, accum, mask, compl. mask, mlir name, comment
    :widths: 20, 10, 10, 10, 10, 10, 20, 20

    mxm             , Y ,   , Y , Y , Y , matrix_multiply,
    vxm             ,   , Y , Y , Y , Y , matrix_multiply,
    mxv             ,   , Y , Y , Y , Y , matrix_multiply,
    eWiseMult       , Y , Y , Y , N , N , intersect,
    eWiseAdd        , Y , Y , Y , N , N , union,
    apply           , Y , Y , Y , N , N , apply,
    apply_Binop1st  , Y , Y , Y , N , N , apply,
    apply_Binop2nd  , Y , Y , Y , N , N , apply,
    select (no val) , Y , Y , Y , N , N , select,
    select (w/ val) , Y , Y , Y , N , N , select,
    reduce_to_scalar, Y , Y , N ,   ,   , reduce_to_scalar,
    reduce_to_vector, Y ,   , Y , N , N , reduce_to_vector,
    transpose       , Y ,   , Y , N , N , transpose,
    kronecker       , N ,   , N , N , N ,,
    diag            , Y , Y ,   ,   ,   , diag,
    assign          , N , N , N , N , N ,,
    col/row assign  , N ,   , N , N , N ,,
    subassign       , N , N , N , N , N ,, GxB
    assign scalar many, Y , Y , Y , Y , Y ,apply/uniform_complement, custom
    extract         , N , N , N , N , N ,,
    col extract     , N ,   , N , N , N ,,
    set element     , N , N ,   ,   ,   ,,
    extract element , N , N ,   ,   ,   ,,
    remove element  , N , N ,   ,   ,   ,,
    build           , Y , Y ,   ,   ,   ,from_coo,
    clear           , N , N ,   ,   ,   ,,
    dup             , Y , Y ,   ,   ,   , dup,
    size/nrows/ncols, Y , Y ,   ,   ,   , size/num_rows/num_cols,
    nvals           , Y , Y ,   ,   ,   , num_vals,
    resize          , N , N ,   ,   ,   ,,
    extractTuples   , Y , Y ,   ,   ,   ,to_coo,
    concat          , N ,   ,   ,   ,   ,, GxB
    split           , N ,   ,   ,   ,   ,, GxB
    isequal         , Y , Y ,   ,   ,   , equal, custom
    vxv/inner       ,   , Y , N ,   ,   , matrix_multiply, custom
    select_rowwise  , Y ,   , Y , N , N , matrix_select_random, custom

.. csv-table:: Supported GraphBLAS operations for updating
    :header: accumulation, mask, replace, is supported
    :widths: 10, 10, 10, 10

    True , True , True , Y
    True , True , False, Y
    True , False,      , Y
    False, True , True , Y
    False, True , False, Y
    False, False,      , Y
