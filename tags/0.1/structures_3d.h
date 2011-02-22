/* structure  for tridiagonal matrix */
typedef struct tridiagonal_matrix_system
{
 double *acof;
 double *bcof;
 double *ccof;
 double *rhs;
}Tds;

/*structure for holding three variables:a,b,c*/
typedef struct tri_variables
{
 double a;
 double b;
 double c;
}Tri_abc;

/*structure for holding momentum(velocity) equation variables */
typedef struct momentum_equation
{
 int nx_max,ny_max,nz_max;
 int istart,iend,jstart,jend,kstart,kend;
 double ***new,***old; //new & old time step values
 double *** pg; //pressure gradient
 double ***explicit_n,***explicit_nm1; // convective terms
 Tri_abc ***Ax,***Ay,***Az; // diffusion operators in x&y-dir,
  
}Velo;

/*structure for holding scalar equation variables at cell center*/
typedef struct concentration_equation
{
 int nx_max,ny_max,nz_max;
 int istart,iend,jstart,jend,kstart,kend;
 double ***new,***old; //new & old time step values
 double ***explicit_n,***explicit_nm1; // convective terms
 Tri_abc ***Ax,***Ay,***Az; // diffusion operators in x&y-dir,
  
}Conc;

/*structure for holding viscosity at both nodal points & cell center*/
typedef struct viscosity_cellcenter_nodalpoints
{
 int nx_max,ny_max,nz_max;
 double ***cell; //value at cell center
 double ***xy;  // value at nodal point in x-y plane
 double ***xz;  // value at nodal point in x-z plane
 double ***yz;  // value at nodal point in y-z plane
  
}Visc;

typedef struct Grid_data
{
 int istart,iend,jstart,jend,kstart,kend;
 int nx_cell,ny_cell,nz_cell;
 int nx_max,ny_max,nz_max;
 int nmax_xyz,nx_global_offset;
 double dx,dy,dz;
 double *X,*Y,*Z;    // nodal points
 double *Xc,*Yc,*Zc; // cell center points
 
}Grid;

