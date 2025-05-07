#include <Python.h>

#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>

//#include "tricub.h"

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


static PyObject* python_ismonotonic(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    return NULL;
}


static PyObject* python_isregular(PyObject* self, PyObject* args)
{    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */
    return NULL;
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