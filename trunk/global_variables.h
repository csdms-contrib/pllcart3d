/*input parameters */
int Nx_cell_global,Ny_cell_global,Nz_cell_global;
int Nx_ghost,Ny_ghost,Nz_ghost;
int Initial,Ugrid,Vdiff;
int Niter_print;
double X1,X2,Y1,Y2,Z1,Z2;
double Re,Pe,Sc,Vratio,Fx,Fy,Inf_pos,Inf_thick;
double CFLmax,Timemax,Timestep,Out_inter,Out_mid;
/* Global calculated variables*/
double PI;
double M,Dratio,Vnorm;

int Wavpert, Subharm, Nmvp;
double Ampli, Lamb, Mum, Lambda;

/*coefficients of RK3-CN*/
double RK1[3] = {(8.0/15.0),(5.0/12.0),(3.0/4.0)};
double RK2[3] = {(0.0),(-17.0/60.0),(-5.0/12.0)};
