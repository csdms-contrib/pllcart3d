/* Global functions */
void input();
void grid_setup(Grid *);
void simulation_setup(Grid*,Velo *,Velo *,Velo *,Conc *);
void mesh_generation(Grid *);
void initial_condition(Grid*,Conc*,Velo*);
void concentration(double ,double ,double ,double,Grid *,double ***,double ***,double ***,Conc *);
double slope_parameter(Grid *grid,double ***cn);
void calculate_viscosity(Grid *,double ***,Visc *);
void calculate_viscosity_nmvp(Grid *grid,double ***cn,Visc *mu,double cm);
void xmom(double,double,double,double,Grid *,Visc *,Visc*,double ***,double ***,double ***,Velo *);
void ymom(double,double,double,double,Grid *,Visc *,Visc*,double ***,double ***,double ***,Velo *);
void zmom(double,double,double,double,Grid *,Visc *,Visc*,double ***,double ***,Velo *);
void A_conc(Grid *,Conc *);
void A_umom(Grid *,Visc *,Velo *);
void A_vmom(Grid *,Visc *,Velo *);
void A_wmom(Grid *,Visc *,Velo *);
void divergence(Grid *,double ***,double ***,double ***,double ***);
void phisolver_dct(double ,double ,double ,Grid *,double***,double ***);
void velo_correction(double ,double ,double,Grid *,double ***,Velo*,Velo *,Velo *);
void update_veloghost_points(Grid *, Velo *,Velo *,Velo *);
void update_ghost_points(Grid *, double ***phi,Velo *,Velo *,Velo *);
void update_velo_boundary(Grid *grid,Velo *u,Velo *v,Velo *w);

void update_phi_ghosts(Grid *grid,double ***phi);
void pressure_gradient(double ,double ,double ,Grid *,double ***,Velo*,Velo *,Velo *);
void write_to_file(int,Grid*,double***,double***,double***,double***); 
double find_cfl(Grid*,double***,double***,double***);
void intermediate(int*,int*,double*,double*,Grid*,Conc*,Velo*,Velo*,Velo*);

void shift_conc(Grid *grid,Conc *c);
void spline(double *x, double *xx, double *y, double *yy, Grid *grid);
void print_to_file(double *y, Grid *grid, int fno);

void midfield(int,int,double,double,double,double,Grid*,Conc*,Velo*,Velo*,Velo*);
double bulk_velo(Grid*,double***);
void update_old_conc(Conc *);
void update_old_velo(Velo *,Velo *,Velo *);
void update_old_visc(Visc *,Visc *);
double calculate_weno(double*,int,int);
void find_max(Grid*,Velo*,Velo*,Velo*,Conc*);
int tridiag_1rhs(int Nmax,double *a,double *b,double *c,double *rhs);

void find_wmax(double tval,Grid *grid,Velo *w);
void conc_average(Grid *grid, double ***cn, double tval);
