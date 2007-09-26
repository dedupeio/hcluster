#include "cluster.h"
#include "Python.h"
#include <numpy/arrayobject.h>
#include <stdio.h>

PyObject *chopmin_ns_ij_wrapper(PyObject *self, PyObject *args);
PyObject *chopmin_ns_i_wrapper(PyObject *self, PyObject *args);
PyObject *chopmins_wrapper(PyObject *self, PyObject *args);

extern PyObject *cluster_wrapper(PyObject *self, PyObject *args) {
  int method, n;
  PyArrayObject *dm, *Z;
  distfunc *df;
  if (!PyArg_ParseTuple(args, "O!O!ii",
			&PyArray_Type, &dm,
			&PyArray_Type, &Z,
			&n,
			&method)) {
    return 0;
  }
  else {
    switch (method) {
    case CPY_LINKAGE_SINGLE:
      df = dist_single;
      break;
    case CPY_LINKAGE_COMPLETE:
      df = dist_complete;
      break;
    default:
      /** Report an error. */
      df = 0;
      break;
    }
    linkage((double*)dm->data, (double*)Z->data, n, 1, df);
  }
  return Py_BuildValue("d", 0.0);
}

extern PyObject *chopmin_ns_ij_wrapper(PyObject *self, PyObject *args) {
  int mini, minj, n;
  PyArrayObject *row;
  if (!PyArg_ParseTuple(args, "O!iii",
			&PyArray_Type, &row,
			&mini,
			&minj,
			&n)) {
    return 0;
  }
  chopmins_ns_ij((double*)row->data, mini, minj, n);
  return Py_BuildValue("d", 0.0);
}

extern PyObject *chopmin_ns_i_wrapper(PyObject *self, PyObject *args) {
  int mini, n;
  PyArrayObject *row;
  if (!PyArg_ParseTuple(args, "O!ii",
			&PyArray_Type, &row,
			&mini,
			&n)) {
    return 0;
  }
  chopmins_ns_i((double*)row->data, mini, n);
  return Py_BuildValue("d", 0.0);
}

extern PyObject *chopmins_wrapper(PyObject *self, PyObject *args) {
  int mini, minj, n;
  PyArrayObject *row;
  if (!PyArg_ParseTuple(args, "O!iii",
			&PyArray_Type, &row,
			&mini,
			&minj,
			&n)) {
    return 0;
  }
  chopmins((int*)row->data, mini, minj, n);
  return Py_BuildValue("d", 0.0);
}

extern PyObject *to_squareform_from_vector(PyObject *self, PyObject *args) {
  PyArrayObject *_M, *_v;
  int n;
  double *v, *M;
  if (!PyArg_ParseTuple(args, "O!O!",
			&PyArray_Type, &_M,
			&PyArray_Type, &_v)) {
    return 0;
  }
  else {
    M = (double*)_M->data;
    v = (double*)_v->data;
    n = _M->dimensions[0];
    dist_to_squareform_from_vector(M, v, n);
  }
  return Py_BuildValue("d", 0.0);
}

extern PyObject *to_vector_from_squareform(PyObject *self, PyObject *args) {
  PyArrayObject *_M, *_v;
  int n;
  double *v, *M;
  if (!PyArg_ParseTuple(args, "O!O!",
			&PyArray_Type, &_M,
			&PyArray_Type, &_v)) {
    return 0;
  }
  else {
    M = (double*)_M->data;
    v = (double*)_v->data;
    n = _M->dimensions[0];
    dist_to_vector_from_squareform(M, v, n);
  }
  return Py_BuildValue("d", 0.0);
}

static PyMethodDef _clusterWrapMethods[] = {
  {"cluster_impl", cluster_wrapper, METH_VARARGS},
  {"chopmins_ns_ij", chopmin_ns_ij_wrapper, METH_VARARGS},
  {"chopmins_ns_i", chopmin_ns_i_wrapper, METH_VARARGS},
  {"chopmins", chopmins_wrapper, METH_VARARGS},
  {"to_squareform_from_vector", to_squareform_from_vector, METH_VARARGS},
  {"to_vector_from_squareform", to_vector_from_squareform, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

void init_cluster_wrap()  {
  (void) Py_InitModule("_cluster_wrap", _clusterWrapMethods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}

