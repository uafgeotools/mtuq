
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>

// function declaration
static PyObject *misfit(PyObject *self, PyObject *args); 

// noilerplate methods list
static PyMethodDef methods[] = {
  { "misfit", misfit, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

// boilerplate module initialization
PyMODINIT_FUNC initext1(void) {
  (void) Py_InitModule("ext1", methods);
  import_array();
}


//
// array access macros
//
#define data(i0,i1,i2)\
    (*(npy_float64*)((PyArray_DATA(data)+\
    (i0) * PyArray_STRIDES(data)[0]+\
    (i1) * PyArray_STRIDES(data)[1]+\
    (i2) * PyArray_STRIDES(data)[2])))

#define greens(i0,i1,i2,i3)\
    (*(npy_float64*)((PyArray_DATA(greens)+\
    (i0) * PyArray_STRIDES(greens)[0]+\
    (i1) * PyArray_STRIDES(greens)[1]+\
    (i2) * PyArray_STRIDES(greens)[2]+\
    (i3) * PyArray_STRIDES(greens)[3])))

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
    (*(npy_float64*)((PyArray_DATA(greens_greens) + \
    (i0) * PyArray_STRIDES(greens_greens)[0]+\
    (i1) * PyArray_STRIDES(greens_greens)[1]+\
    (i2) * PyArray_STRIDES(greens_greens)[2]+\
    (i3) * PyArray_STRIDES(greens_greens)[3]+\
    (i4) * PyArray_STRIDES(greens_greens)[4])))

#define sources(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(sources)+\
    (i0) * PyArray_STRIDES(sources)[0]+\
    (i1) * PyArray_STRIDES(sources)[1])))

#define cc(i0)\
    (*(npy_float64*)((PyArray_DATA(cc)+\
    (i0) * PyArray_STRIDES(cc)[0])))

#define results(i0)\
    (*(npy_float64*)((PyArray_DATA(results)+\
    (i0) * PyArray_STRIDES(results)[0])))


//
// misfit function
//
static PyObject *misfit(PyObject *self, PyObject *args) {

  PyArrayObject *data, *greens, *sources;
  PyArrayObject *data_data, *greens_data, *greens_greens;
  npy_float64 dt;
  int NPAD1, NPAD2;
  int verbose;

  int NSRC,NSTA,NC,NT,NG;
  int isrc,ista,ic,it,ig,j1,j2;

  int itmax, itpad, NPAD;
  npy_float64 norm_sum, norm_tmp, cc_max;
  int nd;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!diii",
                        &PyArray_Type, &data,
                        &PyArray_Type, &greens,
                        &PyArray_Type, &sources,
                        &PyArray_Type, &data_data,
                        &PyArray_Type, &greens_data,
                        &PyArray_Type, &greens_greens,
                        &dt,
                        &NPAD1,
                        &NPAD2,
                        &verbose)) {
    return NULL;
  }

  NSRC = (int) PyArray_SHAPE(sources)[0];
  NSTA = (int) PyArray_SHAPE(data)[0];
  NC = (int) PyArray_SHAPE(data)[1];
  NG = (int) PyArray_SHAPE(greens)[2];
  NT = (int) PyArray_SHAPE(data)[2];

  NPAD = (int) NPAD1+NPAD2+1;

  if (verbose>1) {
    printf("number of sources:  %d\n", NSRC);
    printf("number of stations:  %d\n", NSTA);
    printf("number of components:  %d\n", NC);
    printf("number of time samples:  %d\n", NT);
    printf("number of Green's functions:  %d\n", NG);
  }


  // allocate arrays
  nd = 1;
  npy_intp dims_cc[] = {(int)NPAD};
  PyObject *cc = PyArray_SimpleNew(nd, dims_cc, NPY_DOUBLE);

  nd = 2;
  npy_intp dims_results[] = {(int)NSRC, 1};
  PyObject *results = PyArray_SimpleNew(nd, dims_results, NPY_DOUBLE);


  //
  // begin iterating over sources
  //

  for(isrc=0; isrc<NSRC; ++isrc) {

    norm_sum = (npy_float64) 0.;

    for (ista=0; ista<NSTA; ista++) {
      for (ic=0; ic<NC; ic++) {

        //
        // calculate npts_shift
        //
        for (it=0; it<NPAD; it++) {
          cc(it) = (npy_float64) 0.;
        }
        for (ig=0; ig<NG; ig++) {
          for (it=0; it<NPAD; it++) {
              cc(it) += greens_data(ista,ic,ig,it) * sources(isrc,ig);
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

        //
        // calculate L2 norm using ||s - d||^2 = s^2 + d^2 - 2sd
        //
        norm_tmp = 0.;

        // ss
        for (j1=0; j1<NG; j1++) {
          for (j2=0; j2<NG; j2++) {
            norm_tmp += sources(isrc, j1) * sources(isrc, j2) *
                greens_greens(ista,ic,itpad,j1,j2);
          }
        }

        // dd
        norm_tmp += data_data(ista,ic);

        // sd
        for (ig=0; ig<NG; ig++) {
          norm_tmp -= 2.*greens_data(ista,ic,ig,itpad) * sources(isrc, ig); 
        }

        if (1==1) {
            // L2 norm
            norm_sum += norm_tmp*dt;
        }
        else if (1==0) {
            // hybrid L1-L2 norm
            norm_sum += pow(norm_tmp*dt, 0.5);
        }

      }
    }
    results(isrc) = norm_sum;
  }


  //Py_RETURN_NONE;
  return results;
}

