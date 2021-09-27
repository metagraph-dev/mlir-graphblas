Supported GraphBLAS Operations
==============================

.. csv-table:: Supported GraphBLAS operations
    :header: Operation, Matrix, Vector, accum, mask, complemented mask, comment
    :widths: 20, 10, 10, 10, 10, 10, 30

    mxm             , Y ,   , Y , Y , Y ,
    vxm             ,   , Y , Y , Y , Y ,
    mxv             ,   , Y , Y , Y , Y ,
    eWiseMult       , Y , Y , Y , N , N ,
    eWiseAdd        , Y , Y , Y , N , N ,
    apply           , Y , Y , Y , N , N ,
    apply_Binop1st  , Y , Y , Y , N , N ,
    apply_Binop2nd  , Y , N , Y , N , N ,
    select (no val) , Y , N , Y , N , N ,
    select (w/ val) , Y , N , Y , N , N ,
    reduce_to_scalar, Y , Y , N ,   ,   ,
    reduce_to_vector, Y ,   , Y , N , N ,
    transpose       , Y ,   , Y , N , N ,
    kronecker       , N ,   , Y , N , N ,
    diag            , Y , Y ,   ,   ,   ,
    assign          , N , N , N , N , N ,
    col/row assign  , N ,   , N , N , N ,
    subassign       , N , N , N , N , N , GxB
    assign scalar many, N , N , N , N , N ,
    subassign scalar many, N , N , N , N , N , GxB
    extract         , N , N , N , N , N ,
    col extract     , N ,   , N , N , N ,
    set element     , N , N ,   ,   ,   ,
    extract element , N , N ,   ,   ,   ,
    remove element  , N , N ,   ,   ,   ,
    build           , N , N ,   ,   ,   ,
    clear           , N , N ,   ,   ,   ,
    dup             , Y , Y ,   ,   ,   ,
    size/ncols/nrows, Y , Y ,   ,   ,   ,
    nvals           , Y , Y ,   ,   ,   ,
    resize          , N , N ,   ,   ,   ,
    extractTuples   , N , N ,   ,   ,   ,
    concat          , N ,   ,   ,   ,   , GxB
    split           , N ,   ,   ,   ,   , GxB
    isequal         , Y , Y ,   ,   ,   , custom
    vxv/inner       ,   , Y , N ,   ,   , custom
    argmin/argmax   , Y , Y , Y , N , N , custom
    select_rows("random")   , Y ,   , Y , N , . , custom
    select("random_weighted")   , Y , N , Y , N , N , custom

.. csv-table:: Supported GraphBLAS operations for updating
    :header: accumulation, mask, replace, is supported
    :widths: 10, 10, 10, 10

    True , True , True , N
    True , True , False, N
    True , False,      , Y
    False, True , True , N
    False, True , False, N
    False, False,      , Y
