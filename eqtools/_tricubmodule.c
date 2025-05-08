#include <Python.h>

#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>

#include "_tricub.h"

/*****************************************************************
 
    This file is part of the eqtools package.

    EqTools is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqTools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqTools.  If not, see <http://www.gnu.org/licenses/>.

    Copyright 2025 Ian C. Faust

******************************************************************/

static PyArrayObject* array_check(PyObject* arg, int ndim)
{   /* Check in numpy array, dtype is double, and if the number of dimensions of
    * the array is correct, then return the numpy C-contiguous array. Otherwise,
    * raise a specified Python error and return NULL.
    */
    PyArrayObject* input;

    if (!((arg) && PyArray_Check(arg))){
        PyErr_SetString(PyExc_TypeError, "Input is not a numpy.ndarray subtype");
        return NULL;
    }

    input = (PyArrayObject*)arg;

    if (PyArray_NDIM(input) != ndim){
        PyErr_SetString(PyExc_TypeError, "array has incorrect dimensions");
        return NULL;
    }

    if (PyArray_TYPE(input) != NPY_DOUBLE){
        PyErr_SetString(PyExc_TypeError, "array must be dtype double");
        return NULL;
    }

    if(!PyArray_ISCARRAY_RO(input)) input = PyArray_GETCONTIGUOUS(arg);

    return input;
}

static PyArrayObject* scalar_check(PyObject* arg){
    PyArrayObject* input;


    if (!((arg) && PyArray_CheckScalar(arg))){
        PyErr_SetString(PyExc_TypeError, "Input is not a numpy scalar");
        return NULL;
    }

    input = (PyArrayObject*)arg;

    if (PyArray_TYPE(input) != NPY_INT){
        PyErr_SetString(PyExc_TypeError, "scalar must be dtype int");
        return NULL;
    }

    return input;
}


inline static void parse_input(PyObject* args,
                        double** x0,
                        double** x1,
                        double** x2,
                        double** f,
                        double** fx0,
                        double** fx1,
                        double** fx2,
                        int* ix0,
                        int* ix1,
                        int* ix2,
                        int* ix,
                        int* d0,
                        int* d1,
                        int* d2)
{
    PyObject *x0obj, *x1obj, *x2obj, *fobj, *fx0obj, *fx1obj, *fx2obj;
    PyObject *dobj0 = NULL, *dobj1 = NULL, *dobj2 = NULL;
    PyArg_ParseTuple(args, "O!O!O!O!O!O!O!|O!O!O!");



    if(dobj0) PyArray_ScalarAsCtype(check_scalar(dojb0), (void*)d0);
    if(dobj1) PyArray_ScalarAsCtype(check_scalar(dobj1), (void*)d1);
    if(dobj2) PyArray_ScalarAsCtype(check_scalar(dobj2), (void*)d2);

}


static PyObject* python_reg_ev(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    int ix0, ix1, ix2, ix, d0, d1, d2; // d0, d1, d2 are unused
    double *x0, *x1, *x2, *f, *fx0, *fx1, *fx2;
    parse_input(args, &x0, &x1, &x2, &f, &fx0, &fx1, &fx2, &ix0, &ix1, &ix2, &ix, &d0, &d1, &d2);
    reg_ev(val, x0, x1, x2, f, fx0, fx1, fx2, ix0, ix1, ix2, ix);

}


static PyObject* python_reg_ev_full(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    int ix0, ix1, ix2, ix, d0, d1, d2;
    double *x0, *x1, *x2, *f, *fx0, *fx1, *fx2;
    parse_input(args, &x0, &x1, &x2, &f, &fx0, &fx1, &fx2, &ix0, &ix1, &ix2, &ix, &d0, &d1, &d2);
    reg_ev_full(val, x0, x1, x2, f, fx0, fx1, fx2, ix0, ix1, ix2, ix, d0, d1, d2);
}


static PyObject* python_nonreg_ev(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */

    int ix0, ix1, ix2, ix, d0, d1, d2; // d0, d1, d2 are unused
    double *x0, *x1, *x2, *f, *fx0, *fx1, *fx2;
    parse_input(args, &x0, &x1, &x2, &f, &fx0, &fx1, &fx2, &ix0, &ix1, &ix2, &ix, &d0, &d1, &d2);
    nonreg_ev(val, x0, x1, x2, f, fx0, fx1, fx2, ix0, ix1, ix2, ix);
}


static PyObject* python_nonreg_ev_full(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    int ix0, ix1, ix2, ix, d0, d1, d2;
    double *x0, *x1, *x2, *f, *fx0, *fx1, *fx2;
    parse_input(args, &x0, &x1, &x2, &f, &fx0, &fx1, &fx2, &ix0, &ix1, &ix2, &ix, &d0, &d1, &d2);
    nonreg_ev_full(val, x0, x1, x2, f, fx0, fx1, fx2, ix0, ix1, ix2, ix, d0, d1, d2);
}


static PyObject* python_ev(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    int ix0, ix1, ix2, ix, d0, d1, d2; // d0, d1, d2 are unused
    double *x0, *x1, *x2, *f, *fx0, *fx1, *fx2;
    parse_input(args, &x0, &x1, &x2, &f, &fx0, &fx1, &fx2, &ix0, &ix1, &ix2, &ix, &d0, &d1, &d2);
    ev(val, x0, x1, x2, f, fx0, fx1, fx2, ix0, ix1, ix2, ix)
}


static PyObject* python_ismonotonic(PyObject* self, PyObject* arg)
{
    PyArrayObject* input;
    input = array_check(arg, 1);
    /* if NULL, python error is raised */
    if(!(input)) return NULL;
    
    int ix;
    double* data;

    ix = PyArray_DIM(input, 0);
    data = (double*) PyArray_DATA(input);
    return PyBool_FromLong(ismonotonic(data, ix));
}


static PyObject* python_isregular(PyObject* self, PyObject* arg)
{
    PyArrayObject* input;
    input = array_check(arg, 1);
    /* if NULL, python error is raised */
    if(!(input)) return NULL;
    
    int ix;
    double* data;

    ix = PyArray_DIM(input, 0);
    data = (double*) PyArray_DATA(input);
    return PyBool_FromLong(isregular(data, ix));
}


static PyMethodDef TricubMethods[] = {
    {"reg_ev", python_reg_ev, METH_VARARGS, ""},
    {"reg_ev_full", python_reg_ev_full, METH_VARARGS, ""},
    {"nonreg_ev", python_nonreg_ev, METH_VARARGS, ""},
    {"nonreg_ev_full", python_nonreg_ev_full, METH_VARARGS, ""},
    {"ev", python_ev, METH_VARARGS, ""},
    {"ismonotonic", python_ismonotonic, METH_O, ""},
    {"isregular", python_isregular, METH_O, ""},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};


static struct PyModuleDef _tricubStruct = {
    PyModuleDef_HEAD_INIT,
    "_tricub",
    "",
    -1,
    TricubMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

/* Module initialization */
PyObject *PyInit__tricub(void)
{
    import_array();
    return PyModule_Create(&_tricubStruct);
}