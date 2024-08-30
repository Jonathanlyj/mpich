/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

static void handle_error(int errcode, const char *str)
{
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    fprintf(stderr, "%s: %s\n", str, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define MPI_CHECK(fn) { int errcode; errcode = (fn); if (errcode != MPI_SUCCESS) handle_error(errcode, #fn); }


/* The file name is taken as a command-line argument. */

/* Measures the I/O bandwidth for writing/reading a 3D
   block-distributed array to a file corresponding to the global array
   in row-major (C) order.
   Note that the file access pattern is noncontiguous.

   Array size 128^3. For other array sizes, change array_of_gsizes below.*/


int main(int argc, char **argv)
{
    MPI_Datatype newtype;
    int i, ndims, array_of_gsizes[3], array_of_distribs[3];
    int order, nprocs, len, *d_buf, mynod, item_count;
    MPI_Count bufcount;
    int array_of_dargs[3], array_of_psizes[3];
    MPI_File fh;
    MPI_Status status;
    double stim, write_tim, new_write_tim, write_bw;
    double read_tim, new_read_tim, read_bw;
    char *filename;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynod);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

/* process 0 takes the file name as a command-line argument and
   broadcasts it to other processes */
    if (!mynod) {
        i = 1;
        while ((i < argc) && strcmp("-f", *argv)) {
            i++;
            argv++;
        }
        if (i >= argc) {
            fprintf(stderr, "\n*#  Usage: coll_perf -f filename\n\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        argv++;
        len = (int) strlen(*argv);
        filename = (char *) malloc(len + 1);
        strcpy(filename, *argv);
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(filename, len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        filename = (char *) malloc(len + 1);
        MPI_Bcast(filename, len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    }


    ndims = 3;
    order = MPI_ORDER_C;

    array_of_gsizes[0] = 128 * 10; //default is 17 but LEIA can only takes up to 10
    array_of_gsizes[1] = 128 * 9;
    array_of_gsizes[2] = 128 * 11;

    array_of_distribs[0] = MPI_DISTRIBUTE_BLOCK;
    array_of_distribs[1] = MPI_DISTRIBUTE_BLOCK;
    array_of_distribs[2] = MPI_DISTRIBUTE_BLOCK;

    array_of_dargs[0] = MPI_DISTRIBUTE_DFLT_DARG;
    array_of_dargs[1] = MPI_DISTRIBUTE_DFLT_DARG;
    array_of_dargs[2] = MPI_DISTRIBUTE_DFLT_DARG;

    for (i = 0; i < ndims; i++)
        array_of_psizes[i] = 0;
    MPI_Dims_create(nprocs, ndims, array_of_psizes);

    MPI_CHECK(MPI_Type_create_darray(nprocs, mynod, ndims, array_of_gsizes,
                                     array_of_distribs, array_of_dargs,
                                     array_of_psizes, order, MPI_INT, &newtype));
    MPI_CHECK(MPI_Type_commit(&newtype));

    MPI_CHECK(MPI_Type_size_x(newtype, &bufcount));
    if (bufcount / sizeof(int) >= INT_MAX) {
        fprintf(stderr, "datatype too large: update for MPI-4 support");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* ok to cast: checked for overflow above */
    item_count = (int) (bufcount / sizeof(int));
    CUDA_CHECK(cudaMalloc((void*)&d_buf, item_count * sizeof(int)));
    
/* to eliminate paging effects, do the operations once but don't time
   them */

    MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, filename,
                            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));
    MPI_CHECK(MPI_File_set_view(fh, 0, MPI_INT, newtype, "native", MPI_INFO_NULL));
    MPI_CHECK(MPI_File_write_all(fh, d_buf, item_count, MPI_INT, &status));
    // MPI_CHECK(MPI_File_seek(fh, 0, MPI_SEEK_SET));
    // MPI_CHECK(MPI_File_read_all(fh, buf, item_count, MPI_INT, &status));
    MPI_CHECK(MPI_File_close(&fh));
    MPI_Barrier(MPI_COMM_WORLD);
/* now time write_all */

    MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, filename,
                            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));
    MPI_CHECK(MPI_File_set_view(fh, 0, MPI_INT, newtype, "native", MPI_INFO_NULL));

    MPI_Barrier(MPI_COMM_WORLD);
;
    stim = MPI_Wtime();
    MPI_CHECK(MPI_File_write_all(fh, d_buf, item_count, MPI_INT, &status));
    write_tim = MPI_Wtime() - stim;

    
    MPI_CHECK(MPI_File_close(&fh));

    MPI_Allreduce(&write_tim, &new_write_tim, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


    if (mynod == 0) {
        write_bw =
            (long) (array_of_gsizes[0] * (long) array_of_gsizes[1] * (long) array_of_gsizes[2] *
                    sizeof(int)) / ((new_write_tim) * 1024.0 * 1024.0);
        fprintf(stderr, "Global array size %d x %d x %d integers\n", array_of_gsizes[0],
                array_of_gsizes[1], array_of_gsizes[2]);
        fprintf(stderr,
                "Collective write time = %f sec, Collective write bandwidth = %f Mbytes/sec\n",
                new_write_tim, write_bw);
    }

    MPI_Barrier(MPI_COMM_WORLD);
// /* now time read_all */

//     MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, filename,
//                             MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));
//     MPI_CHECK(MPI_File_set_view(fh, 0, MPI_INT, newtype, "native", MPI_INFO_NULL));

//     MPI_Barrier(MPI_COMM_WORLD);
//     stim = MPI_Wtime();
//     MPI_CHECK(MPI_File_read_all(fh, buf, item_count, MPI_INT, &status));
//     read_tim = MPI_Wtime() - stim;
//     MPI_CHECK(MPI_File_close(&fh));

//     MPI_Allreduce(&read_tim, &new_read_tim, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//     if (mynod == 0) {
//         read_bw =
//             (long) (array_of_gsizes[0] * (long) array_of_gsizes[1] * (long) array_of_gsizes[2] *
//                     sizeof(int)) / (new_read_tim * 1024.0 * 1024.0);
//         fprintf(stderr,
//                 "Collective read time = %f sec, Collective read bandwidth = %f Mbytes/sec\n",
//                 new_read_tim, read_bw);
//     }

    MPI_Type_free(&newtype);

    free(filename);
    cudaFree(d_buf);

    MPI_Finalize();
    return 0;
}
