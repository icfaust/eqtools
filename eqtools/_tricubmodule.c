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

static PyObject* python_reg_ev(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    return NULL;
}


static PyObject* python_reg_ev_full(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    return NULL;
}


static PyObject* python_nonreg_ev(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    return NULL;
}

static PyObject* python_nonreg_ev_full(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    return NULL;
}


static PyObject* python_ev(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    return NULL;
}


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