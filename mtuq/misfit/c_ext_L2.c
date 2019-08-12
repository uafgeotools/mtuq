
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>

// function declaration
static PyObject *misfit(PyObject *self, PyObject *args); 

// boilerplate methods list
static PyMethodDef methods[] = {
  { "misfit", misfit, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL }
};

// boilerplate module initialization
PyMODINIT_FUNC initc_ext_L2(void) {
  (void) Py_InitModule("c_ext_L2", methods);
  import_array();
}


//
// array access macros
//
#define data_data(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(data_data)+\
    (i0) * PyArray_STRIDES(data_data)[0]+\
    (i1) * PyArray_STRIDES(data_data)[1])))

#define greens_data(i0,i1,i2,i3)\
    (*(npy_float64*)((PyArray_DATA(greens_data)+\
    (i0) * PyArray_STRIDES(greens_data)[0]+\
    (i1) * PyArray_STRIDES(greens_data)[1]+\
    (i2) * PyArray_STRIDES(greens_data)[2]+\
    (i3) * PyArray_STRIDES(greens_data)[3])))

#define greens_greens(i0,i1,i2,i3,i4)\
    (*(npy_float64*)((PyArray_DATA(greens_greens)+\
    (i0) * PyArray_STRIDES(greens_greens)[0]+\
    (i1) * PyArray_STRIDES(greens_greens)[1]+\
    (i2) * PyArray_STRIDES(greens_greens)[2]+\
    (i3) * PyArray_STRIDES(greens_greens)[3]+\
    (i4) * PyArray_STRIDES(greens_greens)[4])))

#define sources(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(sources)+\
    (i0) * PyArray_STRIDES(sources)[0]+\
    (i1) * PyArray_STRIDES(sources)[1])))

#define groups(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(groups)+\
    (i0) * PyArray_STRIDES(groups)[0]+\
    (i1) * PyArray_STRIDES(groups)[1])))

#define mask(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(mask)+\
    (i0) * PyArray_STRIDES(mask)[0]+\
    (i1) * PyArray_STRIDES(mask)[1])))

#define results(i0)\
    (*(npy_float64*)((PyArray_DATA(results)+\
    (i0) * PyArray_STRIDES(results)[0])))

#define cc(i0)\
    (*(npy_float64*)((PyArray_DATA(cc)+\
    (i0) * PyArray_STRIDES(cc)[0])))



//
// misfit function
//
static PyObject *misfit(PyObject *self, PyObject *args) {

  PyArrayObject *data_data, *greens_data, *greens_greens;
  PyArrayObject *sources, *groups, *mask;
  int hybrid_norm;
  npy_float64 dt;
  int NPAD1, NPAD2;
  int verbose;

  int NSRC,NSTA,NC,NG,NGRP;
  int isrc,ista,ic,ig,igrp;

  int it, itmax, itpad, j1, j2, NPAD;
  npy_float64 L2_sum, L2_tmp, cc_max;
  int nd;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!idiii",
                        &PyArray_Type, &data_data,
                        &PyArray_Type, &greens_data,
                        &PyArray_Type, &greens_greens,
                        &PyArray_Type, &sources,
                        &PyArray_Type, &groups,
                        &PyArray_Type, &mask,
                        &hybrid_norm,
                        &dt,
                        &NPAD1,
                        &NPAD2,
                        &verbose)) {
    return NULL;
  }

  NSRC = (int) PyArray_SHAPE(sources)[0];
  NSTA = (int) PyArray_SHAPE(mask)[0];
  NC = (int) PyArray_SHAPE(mask)[1];
  NG = (int) PyArray_SHAPE(sources)[1];
  NGRP = (int) PyArray_SHAPE(groups)[0];

  NPAD = (int) NPAD1+NPAD2+1;

  if (verbose>1) {
    printf(" number of groups:  %d\n", NGRP);
    printf(" number of sources:  %d\n", NSRC);
    printf(" number of stations:  %d\n", NSTA);
    printf(" number of components:  %d\n", NC);
    printf(" number of Green's functions:  %d\n\n", NG);
  }


  // allocate arrays
  nd = 1;
  npy_intp dims_cc[] = {(int)NPAD};
  PyObject *cc = PyArray_SimpleNew(nd, dims_cc, NPY_DOUBLE);

  nd = 2;
  npy_intp dims_results[] = {(int)NSRC, 1};
  PyObject *results = PyArray_SimpleNew(nd, dims_results, NPY_DOUBLE);


  //
  //
  // Begin iterating over sources
  //
  //

  for(isrc=0; isrc<NSRC; ++isrc) {

    L2_sum = (npy_float64) 0.;

    for (ista=0; ista<NSTA; ista++) {
      for (igrp=0; igrp<NGRP; igrp++) {

        /*

        Finds the shift between data and synthetics that yields the maximum
        cross-correlation value across all components in the given group,
        subject to the (time_shift_min, time_shift_max) constraint

        */

        for (it=0; it<NPAD; it++) {
          cc(it) = (npy_float64) 0.;
        }

        for (ic=0; ic<NC; ic++) {

          // Skip components not in the group being considered
          if (((int) groups(igrp,ic))==0) {
            continue;
           }

          // Skip traces that have been assigned zero weight
          if (((int) mask(ista,ic))==0) {
              if (verbose>1) {
                if (isrc==0) {
                  printf(" skipping trace: %d %d\n", ista, ic);
                }
              }
              continue;
           }

          // Sum cross-correlations of all components being considered
          for (ig=0; ig<NG; ig++) {
            for (it=0; it<NPAD; it++) {
                cc(it) += greens_data(ista,ic,ig,it) * sources(isrc,ig);
            }
          }
        }
        cc_max = -NPY_INFINITY;
        itmax = 0;
        for (it=0; it<NPAD; it++) {
          if (cc(it) > cc_max) {
            cc_max = cc(it);
            itmax = it;
          }
        }
        itpad = itmax;


        /*

        Calculates L2 norm of difference between data and synthetics
        for all components in the given group

        Rather than computing (s - d) directly for all time samples, we use a
        computational shortcut based on

        ||s - d||^2 = s^2 + d^2 - 2sd

        */
        for (ic=0; ic<NC; ic++) {
          L2_tmp = 0.;

          // Skip components not in the current group being considered
          if (((int) groups(igrp,ic))==0) {
            continue;
          } 

          // Skip traces that have been assigned zero weight
          if (((int) mask(ista,ic))==0) {
              continue;
          }

          // calculate s^2
          for (j1=0; j1<NG; j1++) {
            for (j2=0; j2<NG; j2++) {
              L2_tmp += sources(isrc, j1) * sources(isrc, j2) *
                  greens_greens(ista,ic,itpad,j1,j2);
            }
          }

          // calculate d^2
          L2_tmp += data_data(ista,ic);

          // calculate sd
          for (ig=0; ig<NG; ig++) {
            L2_tmp -= 2.*greens_data(ista,ic,ig,itpad) * sources(isrc, ig); 
          }

          if (hybrid_norm==0) {
              // L2 norm
              L2_sum += dt * L2_tmp;
          }
          else {
              // hybrid L1-L2 norm
              L2_sum += dt * pow(L2_tmp, 0.5);
          }
        }

      }
    }
    results(isrc) = L2_sum;
  }

  return results;

}

