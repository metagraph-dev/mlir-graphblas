Supported GraphBLAS Operations
==============================

.. csv-table:: Supported GraphBLAS operations
    :header: Operation, Matrix, Vector, accum, mask (with replace), mask (w/o replace), comment
    :widths: 20, 10, 10, 10, 10, 10, 30

    mxm             , Y ,   , Y , P , N ,
    vxm             ,   , Y , Y , P , N ,
    mxv             ,   , Y , Y , N , N ,
    eWiseMult       , Y , Y , Y , N , N ,
    eWiseAdd        , Y , Y , Y , N , N ,
    apply           , N , N , Y , N , N ,
    apply_Binop1st  , N , N , Y , N , N ,
    apply_Binop2nd  , Y , N , Y , N , N ,
    select (no val) , Y , N , Y , N , N ,
    select (w/ val) , N , N , Y , N , N ,
    reduce_to_scalar, Y , N , N ,   ,   ,
    reduce_to_vector, N ,   , Y , N , N ,
    transpose       , Y ,   , Y , N , N ,
    kronecker       , N ,   , Y , N , N ,
    diag            , N , N ,   ,   ,   ,
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
    vxv/inner       , N , N ,   ,   ,   , custom
    argmin/argmax   , N , Y , Y , N , N , custom
    select_rows("random")   , N ,   , Y , N , . , custom
    select("random_weighted")   , N , N , Y , N , N , custom
