#ifndef INTEGRATION_ROCK_H
#define INTEGRATION_ROCK_H

typedef void(*func_rock)(int*, double*, double*, double*);

void rock4_integration(double tini, double tend, int n, double *yini, double *y,
                       func_rock fcn, double rtol, double atol, int *info);

void rock4(int *n, double *t, double *tend, double *dt, double *y,
           func_rock fcn, double *atol, double *rtol, double *work, int *iwork, int *idid);

double rho(int *n, double *t, double y);

#endif
