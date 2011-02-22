#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// #include "mpi.h"
// @ Ranger
// #include "/share/apps/pgi7_2/mvapich/1.0.1/include/mpi.h"

// @ beach
// #include "/usr/local/mvapich2/include/mpi.h"
// #include "/usr/local/mpich2-1-1-1p1/include/mpi.h"
// #include "/usr/local/mpich2-1.1/include/mpi.h"
// #include "/usr/local/openmpi-intel/include/mpi.h"
#include "/usr/local/mpich2-gfort-local/include/mpi.h"

// @ teragrid
// #include "/usr/local/apps/fftw301d/include/fftw3.h"
// @ rafael-laptop & Dell (CNSI) & persia
// #include "/usr/local/include/fftw3.h"
// @ caja
// #include "/usr/include/fftw3.h"
// @ Triton
// #include "/opt/pgi/fftw_pgi/include/fftw3.h"
// @ Ranger
// #include "/share/apps/pgi7_2/fftw3/3.1.2/include/fftw3.h"
// @ beach
#include "/usr/local/fftw-3.2.2/include/fftw3.h"

#include "structures_3d.h"
#include "allocate_3d_calloc.h"
#include "fftw_3d.h"
#include "global_variables.h"
#include "global_function_decl.h"

int Rank,Size;

int main(int argc , char * argv[])
{
    int i,j,k,tstep, rkstep, fileno, shift_conc_key = 0;
    double del_min, diff_dt, tval, cfl_dt, dt, rk_gamma, rk_rho, vbulk, outtime, outmid, cavgtime;
    double t1, t2, cm;
    Velo *u,*v,*w;
    Conc *c;
    Visc *mu_old, *mu_new;
    Grid *grid;
    double ***phi, ***div;

    FILE *fp;
  
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Size); 
  
    t1 = MPI_Wtime();
  
    input();
    grid = (Grid*)malloc(sizeof(Grid));
    grid_setup(grid);
  
    u = alloc_velo(grid->nx_max,grid->ny_max,grid->nz_max);
    v = alloc_velo(grid->nx_max,grid->ny_max,grid->nz_max); 
    w = alloc_velo(grid->nx_max,grid->ny_max,grid->nz_max);
    c = alloc_conc(grid->nx_max,grid->ny_max,grid->nz_max);
    mu_old  = alloc_visc(grid->nx_max,grid->ny_max,grid->nz_max);
    mu_new  = alloc_visc(grid->nx_max,grid->ny_max,grid->nz_max);
    phi = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);
    div = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max); 
  
    simulation_setup(grid,u,v,w,c);
    mesh_generation(grid);
  
    plan_dct_3d(grid->ny_cell,grid->nz_cell,grid->nx_cell);
  
    if(Initial == 1) {
        if(Rank == 0) {
            fp = fopen("Outscreen","w");
	    fclose(fp);
	}
        initial_condition(grid,c,u);
        update_ghost_points(grid,phi,u,v,w);
        update_old_conc(c);
        update_old_velo(u,v,w);
        tstep = 0;
        tval = 0;
        fileno = 0;
        outtime = 0;
	outmid = Out_mid;
	cavgtime = 0;
        write_to_file(fileno,grid,u->new,v->new,w->new,c->new);
  
        if(Rank == 0) {
            fp = fopen("filemap.dat","w");
            fprintf(fp,"%d %f\n",fileno,outtime);
            fclose(fp);
	    
	    fp = fopen("cavgy.bin","wb");
            fclose(fp);
	    
	    fp = fopen("cavgz.bin","wb");
            fclose(fp);
	    
	    fp = fopen("cavgyz.bin","wb");
            fclose(fp);
        }
	
        fileno = fileno + 1;
        outtime = tval + Out_inter;
    }
    else {
        intermediate(&tstep,&fileno,&tval,&outtime,grid,c,u,v,w);
        if(Rank == 0){
	    fp = fopen("Outscreen","a");
	    fprintf(fp,"***Starting from intermediate***\n");
	    fclose(fp);
	}
    }
    
    if(Rank == 0){
        fp = fopen("Outscreen","a");
	fprintf(fp,"M=%f Re=%f Pe=%f Vnorm=%f\n",M,Re,Pe,Vnorm);    
        fprintf(fp,"Size = %d\n",Size);
	fclose(fp);
    }
    
    if (Nmvp == 1){
        
	if (Rank == 0){
	    fp = fopen("Outscreen","a");
	    fprintf(fp,"nmvp = 1\n");
	    fclose(fp);
	} 
        cm = slope_parameter(grid,c->new);
	calculate_viscosity_nmvp(grid,c->old,mu_old,cm);
    }	
    else if (Nmvp == 0){
        if (Rank == 0){
	    fp = fopen("Outscreen","a");
	    fprintf(fp,"nmvp = 0\n");
	    fclose(fp);
	} 
        calculate_viscosity(grid,c->old,mu_old);
    }
  
    A_conc(grid,c);
    A_umom(grid,mu_old,u);
    A_vmom(grid,mu_old,v);
    A_wmom(grid,mu_old,w);
  
    del_min = (grid->dx <grid->dy) ? grid->dx: grid->dy;
    del_min = (del_min < grid->dz) ? del_min : grid->dz;
    diff_dt = 0.1*Pe*del_min*del_min; 

    Timestep = (diff_dt < Timestep) ? diff_dt : Timestep;  

    while(tval < Timemax){
     
        tstep = tstep + 1;
        cfl_dt = find_cfl(grid,u->old,v->old,w->old);
        dt = (cfl_dt < Timestep) ? cfl_dt : Timestep;
     
        for(rkstep=0;rkstep<3;rkstep++) {
         
            rk_gamma = RK1[rkstep];
            rk_rho   = RK2[rkstep];
	    tval = tval + (rk_gamma+rk_rho)*dt;

            if(Rank == Size-1) {
	        vbulk = bulk_velo(grid,u->old);
	    }
            concentration(dt,rk_gamma,rk_rho,vbulk,grid,u->new,v->new,w->new,c);
	    
	    if (Nmvp == 1){
	        calculate_viscosity_nmvp(grid,c->new,mu_new,cm);
	    }	
            else if (Nmvp == 0){
                calculate_viscosity(grid,c->new,mu_new);
            }
	    xmom(dt,rk_gamma,rk_rho,vbulk,grid,mu_old,mu_new,c->new,v->old,w->old,u);
	    ymom(dt,rk_gamma,rk_rho,vbulk,grid,mu_old,mu_new,c->new,u->old,w->old,v);
	    zmom(dt,rk_gamma,rk_rho,vbulk,grid,mu_old,mu_new,u->old,v->old,w);
	    update_velo_boundary(grid,u,v,w);
	 
	    divergence(grid,u->new,v->new,w->new,div);
	    phisolver_dct(dt,rk_gamma,rk_rho,grid,div,phi);
	    update_phi_ghosts(grid,phi);
	 
	    velo_correction(dt,rk_gamma,rk_rho,grid,phi,u,v,w);
	    update_veloghost_points(grid,u,v,w);
	 
	    pressure_gradient(dt,rk_gamma,rk_rho,grid,phi,u,v,w);
	    update_old_conc(c);
	    update_old_velo(u,v,w);
	    update_old_visc(mu_old,mu_new);
 
        }
	
	if(tval >= cavgtime){
	    conc_average(grid,c->new,tval);
	    cavgtime = cavgtime + 0.1;
	}    
     
        if(Initial == 0 && Wavpert == 1){
            find_wmax(tval,grid,w);
        }
     
        if((tstep%Niter_print) == 0){
            if(Rank == 0){
	        fp = fopen("Outscreen","a");
	        fprintf(fp,"t = %f\n",tval);
		fclose(fp); 
	    }	
            find_max(grid,u,v,w,c);
        }
          
        if(tval >= outmid){
            if(Rank == 0){
	        fp = fopen("Outscreen","a");
	        fprintf(fp,"\n\n========> WRITE_MIDFIELD:\n========> tstep = %d\n========> tval = %f\n\n",tstep,tval);
		fclose(fp);
	    }	
            midfield(tstep,fileno,tval,outtime,outmid,cavgtime,grid,c,u,v,w);
	    outmid = outmid + Out_mid;
        }
       
        if(tval >= outtime){
            if(Rank == 0){
	        fp = fopen("Outscreen","a");
                fprintf(fp,"\n========> WRITE_TO_FILE: fileno = %d\n========> dt = %f\n========> tstep = %d\n========> tval = %f\n",fileno,dt,tstep,tval);
		fclose(fp);
	    }
	    write_to_file(fileno,grid,u->new,v->new,w->new,c->new);
	    if(Rank ==0) {
	        fp = fopen("filemap.dat","a");
	        fprintf(fp,"%d %f\n",fileno,outtime);
	        fclose(fp);
	    }
	    fileno = fileno + 1;
	    outtime = outtime + Out_inter;
        }
     
        if( tval >= 2 && shift_conc_key == 0 && Wavpert == 1){
	 
	    if(Rank == 0){
	        fp = fopen("Outscreen","a");
	        fprintf(fp,"\n\n=====> ADD WAVE PERTURBATION <=====\n\n");
		fclose(fp);
            }
	    shift_conc(grid,c); 
	    if(Rank == 0 ) {
                fp = fopen("wmax.bin","wb");
                fclose(fp);
            }
	    shift_conc_key = 1;
	 
        }
     
    }
  
    t2 = MPI_Wtime();
    if(Rank == 0){
        fp = fopen("Outscreen","a");
	fprintf(fp,"timetaken : %f   %f\n",t2-t1,(t2-t1)/3600);
	fclose(fp);
    }
    MPI_Finalize();
    return 0;
} 


void input()
{
    int max_len=120;
    char caption[119];
    FILE *ifp;
  
    ifp = fopen("param.input","r");
  
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Nx_cell_global);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Ny_cell_global);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Nz_cell_global);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Nx_ghost);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Ny_ghost);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Nz_ghost);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Initial);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Ugrid);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Niter_print);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Out_mid);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&X1);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&X2);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Y1);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Y2);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Z1);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Z2);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Timemax);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Timestep);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Re);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Pe);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Vratio);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Fx);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Fy);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Inf_pos);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Inf_thick);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&CFLmax);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Out_inter);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Wavpert);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Ampli);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Lamb);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Subharm);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%d \n",&Nmvp);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Mum);
    fgets(caption,max_len,ifp);
    fscanf(ifp,"%lf \n",&Lambda);
  
    fclose(ifp);
  
} 


void grid_setup(Grid *grid)
{
    int i;
   
    grid->nx_cell = (Nx_cell_global + Size - Rank - 1)/Size;
    grid->ny_cell = Ny_cell_global;
    grid->nz_cell = Nz_cell_global;
  
    grid->nx_max = grid->nx_cell + 2*Nx_ghost;
    grid->ny_max = grid->ny_cell + 2*Ny_ghost;
    grid->nz_max = grid->nz_cell + 2*Nz_ghost;

    grid->istart = Nx_ghost;
    grid->iend   = grid->nx_cell +  Nx_ghost - 1;
    grid->jstart = Ny_ghost;
    grid->jend   = grid->ny_cell +  Ny_ghost - 1;
    grid->kstart = Nz_ghost;
    grid->kend   = grid->nz_cell +  Nz_ghost - 1;

    grid->nmax_xyz = (grid->nx_max   > grid->ny_max)? grid->nx_max   : grid->ny_max;
    grid->nmax_xyz = (grid->nmax_xyz > grid->nz_max)? grid->nmax_xyz : grid->nz_max;

    grid->nx_global_offset = 0;
    for(i=0;i< Rank;i++) {
        grid->nx_global_offset += (Nx_cell_global + Size - i - 1)/Size;
    }   
 
    grid->dx = (X2-X1)/(double)Nx_cell_global; 
    grid->dy = (Y2-Y1)/(double)Ny_cell_global; 
    grid->dz = (Z2-Z1)/(double)Nz_cell_global;

    grid->Xc = alloc1d_dble(grid->nx_max);
    grid->Yc = alloc1d_dble(grid->ny_max);
    grid->Zc = alloc1d_dble(grid->nz_max);

    grid->X = alloc1d_dble(grid->nx_max);
    grid->Y = alloc1d_dble(grid->ny_max);
    grid->Z = alloc1d_dble(grid->nz_max);	
   
}

void simulation_setup(Grid *grid,Velo *u,Velo *v,Velo *w,Conc *c)  
{
    
    PI = 4.0*atan(1.0);
  
    // M = log(Vratio); 
    M = Vratio;
 
    // Vnorm = (Vratio > 1.0) ? 1.0 : 0.0 ;
    Vnorm = (Vratio > 0.0) ? 0.0 : 0.0 ;

    u->istart = (Rank ==0) ? grid->istart+1:grid->istart; // exclude left boundary
    u->iend   = grid->iend;
    u->jstart = grid->jstart;
    u->jend   = grid->jend;
    u->kstart = grid->kstart;
    u->kend   = grid->kend;
   
    v->istart = grid->istart;
    v->iend   = grid->iend;
    v->jstart = grid->jstart + 1;  //exclude bottom boundary
    v->jend   = grid->jend;
    v->kstart = grid->kstart;
    v->kend   = grid->kend;
   
    w->istart = grid->istart;
    w->iend   = grid->iend;
    w->jstart = grid->jstart;  
    w->jend   = grid->jend;
    w->kstart = grid->kstart + 1; //exclude side boundary
    w->kend   = grid->kend;
   
    c->istart = grid->istart;
    c->iend   = grid->iend;
    c->jstart = grid->jstart;  
    c->jend   = grid->jend;
    c->kstart = grid->kstart; 
    c->kend   = grid->kend;
     
}


void mesh_generation(Grid *grid)
{
    int i,j,k;
    double xstart,ystart,zstart;
    FILE *fp;
  
    xstart =  X1 + (grid->nx_global_offset - grid->istart)*grid->dx  + (grid->dx/2.0);  
    for(i=0;i< grid->nx_max;i++) {
        grid->Xc[i] = xstart + i*grid->dx ;
    } 
  
    ystart = Y1 - grid->jstart*grid->dy + (grid->dy/2.0);    
    for(j=0;j< grid->ny_max;j++) {
        grid->Yc[j] = ystart + j*grid->dy;
    }
  
    zstart = Z1 - grid->kstart*grid->dz + (grid->dz/2.0);
    for(k=0;k< grid->nz_max;k++) {
        grid->Zc[k] = zstart + k*grid->dz;
    }
  
    if(Rank == 0){    

        if(Nx_cell_global*Ny_cell_global*Nz_cell_global > 2e7){
	    fp = fopen("mesh.dat","w");
            fprintf(fp,"%d %d %d\n",Nx_cell_global/2,Ny_cell_global/2,Nz_cell_global/2);
            xstart = X1 + (grid->dx/2);
	    
            for(i=0;i< Nx_cell_global;i++) {
	        if(i % 2 == 1) fprintf(fp,"%f\n",(float)(xstart+i*grid->dx));
            }
            for(j=grid->jstart;j<= grid->jend;j++) {
                if(j % 2 == 1) fprintf(fp,"%f\n",(float)grid->Yc[j]);
            }
            for(k=grid->kstart;k<= grid->kend;k++) {
                if(k % 2 == 1) fprintf(fp,"%f\n",(float)grid->Zc[k]);
            }
            fclose(fp);
	}else{
	    fp = fopen("mesh.dat","w");
            fprintf(fp,"%d %d %d\n",Nx_cell_global,Ny_cell_global,Nz_cell_global);
            xstart = X1 + (grid->dx/2);
            for(i=0;i< Nx_cell_global;i++) {
                fprintf(fp,"%f\n",(float)(xstart+i*grid->dx));
            }
            for(j=grid->jstart;j<= grid->jend;j++) {
                fprintf(fp,"%f\n",(float)grid->Yc[j]);
            }
            for(k=grid->kstart;k<= grid->kend;k++) {
                fprintf(fp,"%f\n",(float)grid->Zc[k]);
            }
            fclose(fp);
	}
    }
  
}

void initial_condition(Grid *grid,Conc *c,Velo *u)
{
    int m, i,j,k;
    double cval, qin;
    double *q_k;
         
    FILE *fp;

    for(i=c->istart-3;i<= c->iend+3;i++) {      /* c->iend + 3 == c->nx_max - 1*/
        cval = 0.5 + 0.5*erf((grid->Xc[i]-Inf_pos)/Inf_thick);
        for(j=c->jstart-1;j<= c->jend+1;j++) {
  	    for(k=c->kstart-1;k<= c->kend+1;k++) {
  	        c->new[i][j][k] =  cval;
  	    }
        }
    }
  
    if(Rank == 0){
        //for(i=u->istart-1;i<= u->iend+1;i++) { 
            i = u->istart-1;
            for(j=u->jstart;j<= u->jend;j++) {  
                for(k=u->kstart;k<= u->kend;k++) {
                    //u->new[i][j][k] = 1;
                    u->new[i][j][k] = 1.5*(1.0 - 4.0*grid->Yc[j]*grid->Yc[j]);
                } 
                u->new[i][j][u->kstart-1] = u->new[i][j][u->kstart];
                u->new[i][j][u->kend+1]   = u->new[i][j][u->kend];   
            }
            for(k=u->kstart-1;k<= u->kend+1;k++) {
                u->new[i][u->jstart-1][k] = -u->new[i][u->jstart][k];
                u->new[i][u->jend+1][k]   = -u->new[i][u->jend][k];
            }
        //}   
    }
    
    if(Rank == 0 ) {
        // inflow flow rate
        q_k  = alloc1d_dble(u->ny_max);
        i = u->istart - 1;
        for(j=u->jstart;j<= u->jend;j++) { //two-dimensional integration 
            qin = 0.0;
	    for(k=u->kstart+1;k<= u->kend;k++) { //integrate along k-plane
	        qin  += 0.5*grid->dz*(u->new[i][j][k] + u->new[i][j][k-1]);
	    }
            q_k[j]  = qin;
        } 
        qin = 0.0;
        for(j=u->jstart+1;j<= u->jend;j++) { //integrate along j-plane
            qin  += 0.5*grid->dy*(q_k[j-1] + q_k[j]);
        } 
	
	qin /= (Y2-Y1)*(Z2-Z1);
	fp = fopen("Outscreen","a");
	fprintf(fp,"inflow flow rate per unit area = %f\n",qin);
	fclose(fp);
	
        free(q_k);
   
    }       
}


void concentration(double dt,double rk_gamma,double rk_rho,double vbulk,Grid *grid,double ***un,double ***vn,double ***wn,Conc *c)
{
    int i,j,k,nx,ny,nz,nsys;
    int index,rboundary,tri_index;
    double fac,conv,diff,axy,axx,axz;
    double rk_alpha,boun_correction;
    Tds *ptr,*tri_ydir,*tri_xdir,*tri_zdir;
    Tri_abc *ptr_abc;
    double ***source;
    void Hc_op(Grid *,double***,double***,double***,Conc*);  
    void exchange_conc_boundary(Conc*);

    rk_alpha = rk_gamma + rk_rho;
  
    Hc_op(grid,un,vn,wn,c);    // defines c->explicit_n[i][j][k]
   
    source = alloc3d_dble(c->nx_max,c->ny_max,c->nz_max);
  
    for(i=c->istart;i<= c->iend;i++) {
        for(j=c->jstart;j<= c->jend;j++) {
            for(k=c->kstart;k<= c->kend; k++) {
                conv = (rk_gamma*c->explicit_n[i][j][k]+rk_rho*c->explicit_nm1[i][j][k]);
	   
	        ptr_abc = &c->Ax[i][j][k];
                axx     = ptr_abc->a*c->old[i-1][j][k] + ptr_abc->b*c->old[i][j][k]+ptr_abc->c*c->old[i+1][j][k];
                ptr_abc = &c->Ay[i][j][k];
                axy     = ptr_abc->a*c->old[i][j-1][k] + ptr_abc->b*c->old[i][j][k]+ptr_abc->c*c->old[i][j+1][k];
                ptr_abc = &c->Az[i][j][k];
                axz     = ptr_abc->a*c->old[i][j][k-1] + ptr_abc->b*c->old[i][j][k]+ptr_abc->c*c->old[i][j][k+1];
       
                diff = rk_alpha*(axx + axy + axz);
	   
                source[i][j][k] = dt*(conv + diff);
            } 
        } 
    }

    fac = 0.5*dt*rk_alpha;
    ny = c->jend - c->jstart + 1;
    tri_ydir = alloc_tds(1,ny);
  
    for(i=c->istart;i<= c->iend;i++) {
        for(k=c->kstart;k<= c->kend; k++) {

            ptr = tri_ydir;
            index = 0;
            for(j=c->jstart;j<= c->jend;j++) {
                ptr_abc = &c->Ay[i][j][k];
                ptr->acof[index] = -fac*ptr_abc->a;
                ptr->bcof[index] = 1.0-fac*ptr_abc->b;
                ptr->ccof[index] = -fac*ptr_abc->c;
                ptr->rhs[index] = source[i][j][k];
 	        index++;
            }

            ptr->bcof[0] += ptr->acof[0];
            ptr->bcof[ny-1] += ptr->ccof[ny-1];

            tridiag_1rhs(ny,ptr->acof,ptr->bcof,ptr->ccof,ptr->rhs);

            for(j=c->jstart;j<= c->jend;j++) {
                source[i][j][k] = ptr->rhs[j-c->jstart];
            } 
        } 
    }
    free_tds(tri_ydir,1,ny);
  
  
    nz = c->kend - c->kstart + 1;
    tri_zdir = alloc_tds(1,nz);

    for(i=c->istart;i<= c->iend;i++) {
        for(j=c->jstart;j<= c->jend;j++) {

    	    ptr = tri_zdir;

    	    index = 0;
    	    for(k=c->kstart;k<= c->kend; k++) {
    	        ptr_abc = &c->Az[i][j][k];
    	        ptr->acof[index] = -fac*ptr_abc->a;
    	        ptr->bcof[index] = 1.0-fac*ptr_abc->b;
    	        ptr->ccof[index] = -fac*ptr_abc->c;
    	        ptr->rhs[index] = source[i][j][k];
    	        index++;
    	    }
    	    ptr->bcof[0] += ptr->acof[0];
    	    ptr->bcof[nz-1] += ptr->ccof[nz-1];

    	    tridiag_1rhs(nz,ptr->acof,ptr->bcof,ptr->ccof,ptr->rhs);

    	    for(k=c->kstart;k<= c->kend; k++) {
    	        source[i][j][k] = ptr->rhs[k-c->kstart];
    	    } 
        }
    }
    free_tds(tri_zdir,1,nz);
 
 
    nsys = ny*nz;
    nx = c->iend - c->istart+1;
    tri_xdir = alloc_tds(nsys,nx);
 
    tri_index = 0;
 
    for(j=c->jstart;j<= c->jend;j++) {
        for(k=c->kstart;k<= c->kend; k++) {

  	    ptr = tri_xdir + tri_index;
  	    tri_index++;
  
  	    index = 0;
  	    for(i=c->istart;i<= c->iend;i++) {
  	        ptr_abc = &c->Ax[i][j][k];
  	        ptr->acof[index] = -fac*ptr_abc->a;
  	        ptr->bcof[index] = 1.0-fac*ptr_abc->b;
  	        ptr->ccof[index] = -fac*ptr_abc->c;
  	        ptr->rhs[index]  = source[i][j][k];
  	        index++;
  	    }
  	    if(Rank ==0) {
  	        ptr->rhs[0] -= ptr->acof[0]*0.0;
  	    }
  	    if(Rank == Size-1) {
  	        rboundary = c->iend + 1;
  	        c->explicit_n[rboundary][j][k] = -vbulk*(c->old[rboundary][j][k]- c->old[rboundary-1][j][k])/grid->dx;
  	        boun_correction = dt*(rk_gamma*c->explicit_n[rboundary][j][k] + rk_rho*c->explicit_nm1[rboundary][j][k]);
  	        c->new[rboundary][j][k]        = c->old[rboundary][j][k] + boun_correction;
  	        ptr->rhs[nx-1] -= ptr->ccof[nx-1]*boun_correction;
  	    }
       }
    }
   
    pll_tridiag(tri_xdir,nsys,nx);
   
    tri_index = 0;
    for(j=c->jstart;j<= c->jend;j++) {
        for(k=c->kstart;k<= c->kend; k++) {
            ptr = tri_xdir + tri_index;
            tri_index++;
            for(i=c->istart;i<= c->iend;i++) {
                c->new[i][j][k] = c->old[i][j][k] + ptr->rhs[i-c->istart];
            }
        }
    }

    for(i=c->istart;i<= c->iend;i++) {
        for(k=c->kstart;k<= c->kend;k++) {
            c->new[i][c->jstart-1][k] = c->new[i][c->jstart][k];
            c->new[i][c->jend+1][k]   = c->new[i][c->jend][k]; 
        }
    }

    for(i=c->istart;i<= c->iend;i++) {
        for(j=c->jstart-1;j<= c->jend+1;j++) { 
            c->new[i][j][c->kstart-1] = c->new[i][j][c->kstart];
            c->new[i][j][c->kend+1]   = c->new[i][j][c->kend]; 
        }
    } 
  
    exchange_conc_boundary(c);
  
    free_tds(tri_xdir,nsys,nx);
    free3d_dble(c->nx_max,c->ny_max,c->nz_max,source);
      
}


double slope_parameter(Grid *grid,double ***cn)
{
    int i,j,k,jj,kk,niter,idx,n,m,idx1,idx2;
    double yjj,zkk,err,alpha,mum,psi0,psi1,etam,cL,cH,cm,errd,errl,a,cavg,eta,psi;
    double mup1,mup2,lambda;
    double *c1d, *croot, *mucell;
    
    FILE *fp;
    
    n      = grid->iend - grid->istart + 1;
    c1d    = (double*)calloc(n,sizeof(double));
    if (Rank == 0){
        croot  = (double*)calloc(n*Size,sizeof(double));
	mucell = (double*)calloc(n*Size,sizeof(double));
    }	
    
    yjj = 0.0;
    zkk = 0.0;
    err = 10.0;

    for(j=grid->jstart;j<=grid->jend;j++){
    	if (fabs(grid->Yc[j] - yjj) < err){
    	    err = fabs(grid->Yc[j] - yjj);
    	    jj = j;
    	}
    }
    
    err = 10.0;
    for(k=grid->kstart;k<=grid->kend;k++){
    	if (fabs(grid->Zc[k] - zkk) < err){
    	    err = fabs(grid->Zc[k] - zkk);
    	    kk = k;
    	}
    }
    
    for(i=grid->istart;i<= grid->iend;i++) {
        c1d[i-grid->istart] = cn[i][jj][kk];
    }    
    
    MPI_Gather(c1d,n,MPI_DOUBLE,croot,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    if (Rank == 0){
    
        alpha = exp(M);
        mum = Mum*alpha;
        psi0 = asin(alpha/mum);
        psi1 = PI - asin(1/mum); 
        etam = (PI/2 - psi0)/(psi1 - psi0);

        cL = 0.1; cH = 0.9;
        cm = 0.5*(cL + cH);
        errd = 100; errl = 1e-5;
        niter = 0;
        while(errd > 1e-5){
	
    	    a = ( (1-cm) - etam )/( (1 - cm)*(etam - 1) );
    	    
	    for(i=0;i<n*Size;i++) {
    	    
    	        cavg = croot[i];     
    	        eta  = ( (1+a)*cavg )/( 1+a*cavg);
    	        psi  = psi0*(1 - eta) + psi1*eta;
    	        mucell[i] = mum*sin(psi);
		
	    }
	    	
	    idx1 = -1;
	    for(i=0;i<n*Size;i++) {	
    	    
		if(idx1 > 0) break;
		if(fabs(croot[i]) > errl) idx1 = i;
		
    	    }
	    
	    idx2 = -1;
	    for(i=n*Size-1;i>=0;i--) {
	    
	        if(idx2 > 0) break;
		if(fabs(croot[i]) < 1 - errl) idx2 = i;
	    }
	    
	    mup1 = (mucell[idx1+1] - mucell[idx1])/(croot[idx1+1] - croot[idx1]);
	    mup2 = (mucell[idx2] - mucell[idx2-1])/(croot[idx2] - croot[idx2-1]);
	    lambda = (mup1 + mup2)/(alpha + 1);
	    
	    if (lambda > Lambda) cH = cm;
	    else cL = cm;
	    
	    cm = 0.5*(cL+cH);
	    errd = fabs(lambda - Lambda);
	    niter++;
	    if (niter > 500){
	        fp = fopen("Outscreen","a");
	        fprintf(fp,"Lambda did not converge in 500 iterations!");
		fclose(fp);
		exit(1);
	    }
        }
	
	
    }
    
    
    MPI_Bcast(&cm,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    if (Rank == 0){
        fp = fopen("Outscreen","a");
        fprintf(fp,"cm = %f\n",cm);
	fclose(fp);
    }
    
    free(c1d);
    if (Rank == 0){
        free(croot);
	free(mucell);
    }	
    return cm;

}


void calculate_viscosity(Grid *grid,double ***cn,Visc *mu)
{
    int i,j,k;
    double cavg;

    /* viscosity at cell centre */
    for(i=grid->istart-1;i<= grid->iend+1;i++) {
        for(j=grid->jstart-1;j<= grid->jend+1;j++) {
            for(k=grid->kstart-1;k<= grid->kend+1;k++) {
                cavg              = cn[i][j][k];     
                mu->cell[i][j][k] = exp(M*(cavg-Vnorm)); // now calculate the new
            }
        }
    }  
    
    /* viscosity at nodal points in x-y plane(Note: x & y indices are from start to end+1*/
    for(i=grid->istart;i<= grid->iend +1;i++) {
        for(j=grid->jstart;j<= grid->jend +1;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
                cavg            = 0.25*(cn[i-1][j-1][k]+cn[i][j-1][k]+cn[i-1][j][k]+cn[i][j][k]);
                mu->xy[i][j][k] = exp(M*(cavg-Vnorm));
            }
        }
    }

    for(i=grid->istart;i<= grid->iend +1;i++) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend +1;k++) {
                cavg            = 0.25*(cn[i-1][j][k-1]+cn[i][j][k-1]+cn[i-1][j][k]+cn[i][j][k]);
                mu->xz[i][j][k] = exp(M*(cavg-Vnorm));
            }
        }
    }

    for(i=grid->istart;i<= grid->iend;i++) {
        for(j=grid->jstart;j<= grid->jend +1 ;j++) {
            for(k=grid->kstart;k<= grid->kend +1;k++) {
                cavg            = 0.25*(cn[i][j-1][k-1]+cn[i][j][k-1]+cn[i][j-1][k]+cn[i][j][k]);
                mu->yz[i][j][k] = exp(M*(cavg-Vnorm));
            }
        }
    }  
       
} 


void calculate_viscosity_nmvp(Grid *grid,double ***cn,Visc *mu,double cm)
{
    int i,j,k;
    double alpha,mum,psi0,psi1,etam,eta,psi,a;
    double cavg;
    
    alpha = exp(M);
    mum = Mum*alpha;
    psi0 = asin(alpha/mum);
    psi1 = PI - asin(1/mum); 
    etam = (PI/2 - psi0)/(psi1 - psi0);
    a = ( (1-cm) - etam )/( (1 - cm)*(etam - 1) );
    
    /* viscosity at cell centre */
    for(i=grid->istart-1;i<= grid->iend+1;i++) {
    	for(j=grid->jstart-1;j<= grid->jend+1;j++) {
    	    for(k=grid->kstart-1;k<= grid->kend+1;k++) {
    		cavg		  = cn[i][j][k];
    		eta = ((1+a)*cavg)/(1+a*cavg);
    		psi = psi1*(1 - eta) + psi0*eta;
    		mu->cell[i][j][k] = mum*sin(psi);
    	    }
    	}
    }
    
    /* viscosity at nodal points in x-y plane(Note: x & y indices are from start to end+1*/
    for(i=grid->istart;i<= grid->iend +1;i++) {
    	for(j=grid->jstart;j<= grid->jend +1;j++) {
    	    for(k=grid->kstart;k<= grid->kend;k++) {
    		cavg		= 0.25*(cn[i-1][j-1][k]+cn[i][j-1][k]+cn[i-1][j][k]+cn[i][j][k]);
    		eta = ((1+a)*cavg)/(1+a*cavg);
    		psi = psi1*(1 - eta) + psi0*eta;
    		mu->xy[i][j][k] = mum*sin(psi);
    	    }
    	}
    }

    for(i=grid->istart;i<= grid->iend +1;i++) {
    	for(j=grid->jstart;j<= grid->jend;j++) {
    	    for(k=grid->kstart;k<= grid->kend +1;k++) {
    		cavg		= 0.25*(cn[i-1][j][k-1]+cn[i][j][k-1]+cn[i-1][j][k]+cn[i][j][k]);
    		eta = ((1+a)*cavg)/(1+a*cavg);
    		psi = psi1*(1 - eta) + psi0*eta;
    		mu->xz[i][j][k] = mum*sin(psi);
    	    }
    	}
    }

    for(i=grid->istart;i<= grid->iend;i++) {
    	for(j=grid->jstart;j<= grid->jend +1 ;j++) {
    	    for(k=grid->kstart;k<= grid->kend +1;k++) {
    		cavg		= 0.25*(cn[i][j-1][k-1]+cn[i][j][k-1]+cn[i][j-1][k]+cn[i][j][k]);
    		eta = ((1+a)*cavg)/(1+a*cavg);
    		psi = psi1*(1 - eta) + psi0*eta;
    		mu->yz[i][j][k] = mum*sin(psi);
    	    }
    	}
    }

}



void xmom(double dt,double rk_gamma,double rk_rho,double vbulk,Grid *grid,Visc *mu_old,
          Visc *mu_new,double ***cn,double ***vold,double ***wold,Velo *u)
{
    int i,j,k,nx,ny,nz,nsys;
    int index, tri_index, rboundary;
    double fac,conv,visc,bodyforce,axy,axx,axz;
    double rk_alpha,cval,boun_correction;
    Tds *ptr,*tri_ydir,*tri_xdir,*tri_zdir;
    Tri_abc *ptr_abc;
    double ***source;
    void Hu_op(Grid *,Visc*,double***,double***,Velo*);  

    rk_alpha = rk_gamma + rk_rho;

    Hu_op(grid,mu_old,vold,wold,u);

    source = alloc3d_dble(u->nx_max,u->ny_max,u->nz_max);
  
    for(i=u->istart;i<= u->iend;i++) {
        for(j=u->jstart;j<= u->jend;j++) {
            for(k=u->kstart;k<= u->kend; k++) {
	        conv = (rk_gamma*u->explicit_n[i][j][k]+rk_rho*u->explicit_nm1[i][j][k]);
	   
                ptr_abc = &u->Ax[i][j][k];
                axx     = ptr_abc->a*u->old[i-1][j][k] + ptr_abc->b*u->old[i][j][k] + ptr_abc->c*u->old[i+1][j][k];
                ptr_abc = &u->Ay[i][j][k];
                axy     = ptr_abc->a*u->old[i][j-1][k] + ptr_abc->b*u->old[i][j][k] + ptr_abc->c*u->old[i][j+1][k];
                ptr_abc = &u->Az[i][j][k];
                axz     = ptr_abc->a*u->old[i][j][k-1] + ptr_abc->b*u->old[i][j][k] + ptr_abc->c*u->old[i][j][k+1];
                visc    = rk_alpha*(axx + axy + axz);
	   
	        cval      = 0.5*(cn[i][j][k]+cn[i-1][j][k]);
                bodyforce = rk_alpha*(-(Fx/Re)*cval - (u->pg[i][j][k]/Re));

                source[i][j][k] = dt*(conv + visc + bodyforce);
            }
        }
    }

    A_umom(grid,mu_new,u); 
    fac = 0.5*dt*rk_alpha;

    ny = u->jend - u->jstart + 1;
    tri_ydir = alloc_tds(1,ny);
  
    for(i=u->istart;i<= u->iend;i++) {
        for(k=u->kstart;k<= u->kend; k++) {
            ptr   = tri_ydir;
            index = 0;
            for(j=u->jstart;j<= u->jend;j++) {
                ptr_abc          = &u->Ay[i][j][k];
                ptr->acof[index] = -fac*ptr_abc->a;
                ptr->bcof[index] = 1.0-fac*ptr_abc->b;
                ptr->ccof[index] = -fac*ptr_abc->c;
                ptr->rhs[index]  = source[i][j][k];
                index++;
            }
	    /*bottom boundary: dirichlet: u = 0*/
            ptr->bcof[0]    -= ptr->acof[0];
            ptr->bcof[ny-1] -= ptr->ccof[ny-1];
     
            tridiag_1rhs(ny,ptr->acof,ptr->bcof,ptr->ccof,ptr->rhs);
     
            for(j=u->jstart;j<= u->jend;j++) {
                source[i][j][k] = ptr->rhs[j-u->jstart];
            } 
        }
    }   
    free_tds(tri_ydir,1,ny);
  

    nz = u->kend - u->kstart + 1;
    tri_zdir = alloc_tds(1,nz);
  
    for(i=u->istart;i<= u->iend;i++) {
        for(j=u->jstart;j<= u->jend;j++) {
            ptr = tri_zdir;
            index = 0;
            for(k=u->kstart;k<= u->kend; k++) {
                ptr_abc          = &u->Az[i][j][k];
                ptr->acof[index] = -fac*ptr_abc->a;
                ptr->bcof[index] = 1.0-fac*ptr_abc->b;
                ptr->ccof[index] = -fac*ptr_abc->c;
                ptr->rhs[index]  = source[i][j][k];
                index++;
            }
	    /*side boundary: symmetry: dudz=0*/
            ptr->bcof[0]    += ptr->acof[0];
            ptr->bcof[nz-1] += ptr->ccof[nz-1];

            tridiag_1rhs(nz,ptr->acof,ptr->bcof,ptr->ccof,ptr->rhs);
     
            for(k=u->kstart;k<= u->kend; k++) {
                source[i][j][k] = ptr->rhs[k-u->kstart];
            } 
        }
    } 
    free_tds(tri_zdir,1,nz);
  
  
    nsys = ny*nz;
    nx = u->iend - u->istart+1;
    tri_xdir = alloc_tds(nsys,nx);
  
    tri_index = 0;
  
    for(j=u->jstart;j<= u->jend;j++) {
        for(k=u->kstart;k<= u->kend; k++) {
        
	    ptr = tri_xdir + tri_index;
	    tri_index++;

            index = 0;
            for(i=u->istart;i<= u->iend;i++) {
                ptr_abc          = &u->Ax[i][j][k];
	        ptr->acof[index] = -fac*ptr_abc->a;
	        ptr->bcof[index] = 1.0-fac*ptr_abc->b;
                ptr->ccof[index] = -fac*ptr_abc->c;
                ptr->rhs[index]  = source[i][j][k];
	        index++;
            }
            if(Rank ==0) {
	        /*left boundary:dirichlet vz = given inlet velo*/
                ptr->rhs[0] -= ptr->acof[0]*0.0;
            }
            if(Rank == Size-1) {
	        /* right boundary:outflow
	        solution explicitly found by convective outflow
	       */
	        rboundary = u->iend + 1;
	        u->explicit_n[rboundary][j][k] = -vbulk*(u->old[rboundary][j][k]- u->old[rboundary-1][j][k])/grid->dx;
                boun_correction = dt*(rk_gamma*u->explicit_n[rboundary][j][k] + rk_rho*u->explicit_nm1[rboundary][j][k]);
	        u->new[rboundary][j][k]        = u->old[rboundary][j][k] + boun_correction;
		/* add the contribution to rhs of linear system */
	        ptr->rhs[nx-1] -= ptr->ccof[nx-1]*boun_correction;
            }
        }
    }
  
    pll_tridiag(tri_xdir,nsys,nx);
  
    tri_index = 0;
    for(j=u->jstart;j<= u->jend;j++) {
        for(k=u->kstart;k<= u->kend; k++) {
            ptr = tri_xdir + tri_index;
            tri_index++;
            for(i=u->istart;i<= u->iend;i++) {
                u->new[i][j][k] = u->old[i][j][k] + ptr->rhs[i-u->istart];
            }  
        }       
    }
  
    free_tds(tri_xdir,nsys,nx);
    free3d_dble(u->nx_max,u->ny_max,u->nz_max,source);
        
}  


void ymom(double dt,double rk_gamma,double rk_rho,double vbulk,Grid *grid,Visc *mu_old,
          Visc *mu_new,double ***cn,double ***uold,double ***wold,Velo *v)
{
    int i,j,k,nx,ny,nz,nsys;
    int index,tri_index,rboundary;
    double fac,conv,visc,bodyforce,ayx,ayy,ayz;
    double rk_alpha,cval,boun_correction;
    Tds *ptr,*tri_ydir,*tri_xdir,*tri_zdir;
    Tri_abc *ptr_abc;
    double ***source;
    void Hv_op(Grid *,Visc*,double***,double***,Velo*);
    
    rk_alpha = rk_gamma + rk_rho;

    Hv_op(grid,mu_old,uold,wold,v);

    source = alloc3d_dble(v->nx_max,v->ny_max,v->nz_max);
    for(i=v->istart;i<= v->iend;i++) {
        for(j=v->jstart;j<= v->jend;j++) {
            for(k=v->kstart;k<= v->kend; k++) {
                conv = (rk_gamma*v->explicit_n[i][j][k]+rk_rho*v->explicit_nm1[i][j][k]);
	   
                ptr_abc = &v->Ax[i][j][k];
                ayx     = ptr_abc->a*v->old[i-1][j][k] + ptr_abc->b*v->old[i][j][k] + ptr_abc->c*v->old[i+1][j][k];
                ptr_abc = &v->Ay[i][j][k];
                ayy     = ptr_abc->a*v->old[i][j-1][k] + ptr_abc->b*v->old[i][j][k] + ptr_abc->c*v->old[i][j+1][k];
                ptr_abc = &v->Az[i][j][k];
                ayz     = ptr_abc->a*v->old[i][j][k-1] + ptr_abc->b*v->old[i][j][k] + ptr_abc->c*v->old[i][j][k+1];
                visc    = rk_alpha*(ayx + ayy + ayz);
	   
                cval      = 0.5*(cn[i][j][k] + cn[i][j-1][k]);
                bodyforce = rk_alpha*(-(Fy/Re)*cval - (v->pg[i][j][k]/Re));

                source[i][j][k] = dt*(conv + visc + bodyforce);
            } 
        }
    }

    A_vmom(grid,mu_new,v);
    fac = 0.5*dt*rk_alpha;

    ny = v->jend - v->jstart + 1;
    tri_ydir = alloc_tds(1,ny);
    for(i=v->istart;i<= v->iend;i++) {
        for(k=v->kstart;k<= v->kend; k++) {
            ptr = tri_ydir;

            index = 0;
            for(j=v->jstart;j<= v->jend;j++) {
                ptr_abc = &v->Ay[i][j][k];
                ptr->acof[index] = -fac*ptr_abc->a;
                ptr->bcof[index] = 1.0-fac*ptr_abc->b;
                ptr->ccof[index] = -fac*ptr_abc->c;
                ptr->rhs[index]  = source[i][j][k];
	        index++;
            }
            /*bottom boundary: dirichlet:v = 0*/
            ptr->rhs[0]    -= ptr->acof[0]*0.0;
            ptr->rhs[ny-1] -= ptr->ccof[ny-1]*0.0;   
    
            tridiag_1rhs(ny,ptr->acof,ptr->bcof,ptr->ccof,ptr->rhs);
    	 
            for(j=v->jstart;j<= v->jend;j++) {
                source[i][j][k] = ptr->rhs[j-v->jstart];
            }        
        }
    }  
    free_tds(tri_ydir,1,ny);
 

    nz = v->kend - v->kstart + 1;
    tri_zdir = alloc_tds(1,nz);
    for(i=v->istart;i<= v->iend;i++) {
        for(j=v->jstart;j<= v->jend;j++) {
            ptr = tri_zdir;

            index = 0;
            for(k=v->kstart;k<= v->kend; k++) {
                ptr_abc = &v->Az[i][j][k];
                ptr->acof[index] = -fac*ptr_abc->a;
                ptr->bcof[index] = 1.0-fac*ptr_abc->b;
                ptr->ccof[index] = -fac*ptr_abc->c;
                ptr->rhs[index] = source[i][j][k];	
	        index++;
            }
	    /*side boundary: symmetry: dvdz=0*/
            ptr->bcof[0]    += ptr->acof[0];
            ptr->bcof[nz-1] += ptr->ccof[nz-1];  

            tridiag_1rhs(nz,ptr->acof,ptr->bcof,ptr->ccof,ptr->rhs);

            for(k=v->kstart;k<= v->kend; k++) {
            source[i][j][k] = ptr->rhs[k-v->kstart];
            }       
        }
    }  
    free_tds(tri_zdir,1,nz);
  
  
    nsys = ny*nz;
    nx = v->iend - v->istart+1;
    tri_xdir = alloc_tds(nsys,nx);
  
    tri_index = 0;

    for(j=v->jstart;j<= v->jend;j++) {
        for(k=v->kstart;k<= v->kend; k++) {
            ptr = tri_xdir + tri_index;
	    tri_index++;

            index = 0;
            for(i=v->istart;i<= v->iend;i++) {
                ptr_abc = &v->Ax[i][j][k];
                ptr->acof[index] = -fac*ptr_abc->a;
                ptr->bcof[index] = 1.0-fac*ptr_abc->b;
                ptr->ccof[index] = -fac*ptr_abc->c;
                ptr->rhs[index]  = source[i][j][k];
	        index++;
            }
	    /*left boundary:dirichlet v = 0*/
            if(Rank ==0) {
                ptr->bcof[0] -= ptr->acof[0];
            }
            if(Rank == Size-1) {
	        rboundary = v->iend + 1;
	        v->explicit_n[rboundary][j][k] = -vbulk*(v->old[rboundary][j][k]- v->old[rboundary-1][j][k])/grid->dx;
                boun_correction = dt*(rk_gamma*v->explicit_n[rboundary][j][k] + rk_rho*v->explicit_nm1[rboundary][j][k]);
	        v->new[rboundary][j][k]        = v->old[rboundary][j][k] + boun_correction;
	        ptr->rhs[nx-1] -= ptr->ccof[nx-1]*boun_correction;
            }
        } 
    }
   
    pll_tridiag(tri_xdir,nsys,nx);
  
    tri_index = 0;
  
    for(j=v->jstart;j<= v->jend;j++) {
        for(k=v->kstart;k<= v->kend; k++) {
            ptr = tri_xdir + tri_index;
            tri_index++;
            for(i=v->istart;i<= v->iend;i++) {
                v->new[i][j][k] = v->old[i][j][k] + ptr->rhs[i-v->istart];
            }  
        }       
    }
     
    free_tds(tri_xdir,nsys,nx);
    free3d_dble(v->nx_max,v->ny_max,v->nz_max,source);
         
} 


void zmom(double dt,double rk_gamma,double rk_rho,double vbulk,Grid *grid,Visc *mu_old,
          Visc *mu_new,double ***uold,double ***vold,Velo *w)
{
    int i,j,k,nx,ny,nz,nsys;
    int index,tri_index,rboundary;
    double fac,conv,visc,bodyforce,ayx,ayy,ayz;
    double rk_alpha,boun_correction;
    Tds *ptr,*tri_ydir,*tri_xdir,*tri_zdir;
    Tri_abc *ptr_abc;
    double ***source;
    void Hw_op(Grid *,Visc*,double***,double***,Velo*);

    rk_alpha = rk_gamma + rk_rho;

    Hw_op(grid,mu_old,uold,vold,w);

    source = alloc3d_dble(w->nx_max,w->ny_max,w->nz_max);
    for(i=w->istart;i<= w->iend;i++) {
        for(j=w->jstart;j<= w->jend;j++) {
            for(k=w->kstart;k<= w->kend; k++) {
                conv = (rk_gamma*w->explicit_n[i][j][k]+rk_rho*w->explicit_nm1[i][j][k]);
       
                ptr_abc = &w->Ax[i][j][k];
                ayx     = ptr_abc->a*w->old[i-1][j][k] + ptr_abc->b*w->old[i][j][k] + ptr_abc->c*w->old[i+1][j][k];
                ptr_abc = &w->Ay[i][j][k];
                ayy     = ptr_abc->a*w->old[i][j-1][k] + ptr_abc->b*w->old[i][j][k] + ptr_abc->c*w->old[i][j+1][k];
                ptr_abc = &w->Az[i][j][k];
                ayz     = ptr_abc->a*w->old[i][j][k-1] + ptr_abc->b*w->old[i][j][k] + ptr_abc->c*w->old[i][j][k+1];
                visc    = rk_alpha*(ayx + ayy + ayz);
	   
                bodyforce = rk_alpha*( -(w->pg[i][j][k]/Re));
	   
                source[i][j][k] = dt*(conv + visc + bodyforce);
            } 
        }
    }

  A_wmom(grid,mu_new,w);
  fac = 0.5*dt*rk_alpha;
  
  ny = w->jend - w->jstart + 1;
  tri_ydir = alloc_tds(1,ny);
  for(i=w->istart;i<= w->iend;i++) {
     for(k=w->kstart;k<= w->kend; k++) {
        ptr = tri_ydir;
        index = 0;
        for(j=w->jstart;j<= w->jend;j++) {
           ptr_abc = &w->Ay[i][j][k];
           ptr->acof[index] = -fac*ptr_abc->a;
           ptr->bcof[index] = 1.0-fac*ptr_abc->b;
           ptr->ccof[index] = -fac*ptr_abc->c;
           ptr->rhs[index]  = source[i][j][k];
	   index++;
        }
        ptr->bcof[0] -= ptr->acof[0];
        ptr->bcof[ny-1] -= ptr->ccof[ny-1];   
    
        tridiag_1rhs(ny,ptr->acof,ptr->bcof,ptr->ccof,ptr->rhs);
    	 
        for(j=w->jstart;j<= w->jend;j++) {
           source[i][j][k] = ptr->rhs[j-w->jstart];
        }        
     }
  }  
  free_tds(tri_ydir,1,ny);
 

  nz = w->kend - w->kstart + 1;
  tri_zdir = alloc_tds(1,nz);
  for(i=w->istart;i<= w->iend;i++) {
     for(j=w->jstart;j<= w->jend;j++) {
        ptr = tri_zdir;
        index = 0;
        for(k=w->kstart;k<= w->kend; k++) {
           ptr_abc = &w->Az[i][j][k];
           ptr->acof[index] = -fac*ptr_abc->a;
           ptr->bcof[index] = 1.0-fac*ptr_abc->b;
           ptr->ccof[index] = -fac*ptr_abc->c;
           ptr->rhs[index]  = source[i][j][k];
	   index++;
        }
        ptr->rhs[0] -= ptr->acof[0]*0;
        ptr->rhs[nz-1] -= ptr->ccof[nz-1]*0.0; 

        tridiag_1rhs(nz,ptr->acof,ptr->bcof,ptr->ccof,ptr->rhs);

        for(k=w->kstart;k<= w->kend; k++) {
           source[i][j][k] = ptr->rhs[k-w->kstart];
        }       
     }
  }  
  free_tds(tri_zdir,1,nz);
  
  
  nsys = ny*nz;
  nx = w->iend - w->istart+1;
  tri_xdir = alloc_tds(nsys,nx);
  
  tri_index = 0;
  
  for(j=w->jstart;j<= w->jend;j++) {
     for(k=w->kstart;k<= w->kend; k++) {
        ptr = tri_xdir + tri_index;
	tri_index++;

        index = 0;
        for(i=w->istart;i<= w->iend;i++) {
           ptr_abc = &w->Ax[i][j][k];
           ptr->acof[index] = -fac*ptr_abc->a;
           ptr->bcof[index] = 1.0-fac*ptr_abc->b;
           ptr->ccof[index] = -fac*ptr_abc->c;
           ptr->rhs[index]  = source[i][j][k];	 
	   index++;
        }
        if(Rank ==0) {
           ptr->bcof[0] -= ptr->acof[0];

        }
        if(Rank == Size-1) {
	   rboundary = w->iend + 1;
	   w->explicit_n[rboundary][j][k] = -vbulk*(w->old[rboundary][j][k]- w->old[rboundary-1][j][k])/grid->dx;
           boun_correction = dt*(rk_gamma*w->explicit_n[rboundary][j][k] + rk_rho*w->explicit_nm1[rboundary][j][k]);
	   w->new[rboundary][j][k]        = w->old[rboundary][j][k] + boun_correction;
	   ptr->rhs[nx-1] -= ptr->ccof[nx-1]*boun_correction;
        }
     }  
  }

  pll_tridiag(tri_xdir,nsys,nx);

  tri_index = 0;
  
  for(j=w->jstart;j<= w->jend;j++) {
     for(k=w->kstart;k<= w->kend; k++) {
        ptr = tri_xdir + tri_index;
        tri_index++;
        for(i=w->istart;i<= w->iend;i++) {
           w->new[i][j][k] = w->old[i][j][k] + ptr->rhs[i-w->istart];
        }  
     }       
  }
    
  free_tds(tri_xdir,nsys,nx);
  free3d_dble(w->nx_max,w->ny_max,w->nz_max,source);

} 


void A_conc(Grid *grid,Conc *c)
{
    int i,j,k;
    double del;
    Tri_abc *ptr;
  
    for(i=c->istart;i<= c->iend;i++) {
        for(j=c->jstart;j<= c->jend;j++) {
            for(k=c->kstart;k<= c->kend;k++) {
                ptr    = &c->Ax[i][j][k];
                del    = 1.0/(Pe*grid->dx*grid->dx);
	        ptr->a = del;
	        ptr->c = del;
	        ptr->b = -(ptr->a + ptr->c);

                ptr = &c->Ay[i][j][k];
	        del    = 1.0/(Pe*grid->dy*grid->dy);
	        ptr->a = del;
	        ptr->c = del;
	        ptr->b = -(ptr->a + ptr->c);

                ptr = &c->Az[i][j][k];
	        del    = 1.0/(Pe*grid->dz*grid->dz);
	        ptr->a = del;
	        ptr->c = del;
	        ptr->b = -ptr->a -ptr->c ;
            }	 
        }
    } 
}  


void A_umom(Grid *grid,Visc *mu,Velo *u)
{
    int i,j,k;
    double del;
    Tri_abc *ptr;
  
    for(i=u->istart;i<= u->iend;i++) {
        for(j=u->jstart;j<= u->jend;j++) {
            for(k=u->kstart;k<= u->kend;k++) {
                ptr    = &u->Ax[i][j][k];
                del    = 2.0/(Re*grid->dx);
                ptr->a = del*mu->cell[i-1][j][k]/grid->dx;
	        ptr->c = del*mu->cell[i][j][k]/grid->dx;
	        ptr->b = -(ptr->a + ptr->c);

                ptr    = &u->Ay[i][j][k];
	        del    = 1.0/(Re*grid->dy);
	        ptr->a = del*mu->xy[i][j][k]/grid->dy;
	        ptr->c = del*mu->xy[i][j+1][k]/grid->dy;
	        ptr->b = -(ptr->a + ptr->c);

                ptr    = &u->Az[i][j][k];
	        del    = 1.0/(Re*grid->dz);
	        ptr->a = del*mu->xz[i][j][k]/grid->dz;
	        ptr->c = del*mu->xz[i][j][k+1]/grid->dz;
	        ptr->b = -ptr->a -ptr->c ;
            }	 
        }
    } 
}  


void A_vmom(Grid *grid,Visc *mu,Velo *v)
{
    int i,j,k;
    double del;
    Tri_abc *ptr;
  
    for(i=v->istart;i<= v->iend;i++) {
        for(j=v->jstart;j<= v->jend;j++) {
            for(k=v->kstart;k<= v->kend;k++) {
                ptr    = &v->Ax[i][j][k];
                del    = 1.0/(Re*grid->dx);
	        ptr->a = del*mu->xy[i][j][k]/grid->dx;
	        ptr->c = del*mu->xy[i+1][j][k]/grid->dx;
	        ptr->b = -ptr->a -ptr->c ;

                ptr    = &v->Ay[i][j][k];
	        del    = 2.0/(Re*grid->dy);
	        ptr->a = del*mu->cell[i][j-1][k]/grid->dy;
	        ptr->c = del*mu->cell[i][j][k]/grid->dy;
	        ptr->b = -ptr->a -ptr->c ;

                ptr    = &v->Az[i][j][k];
                del    = 1.0/(Re*grid->dz);
	        ptr->a = del*mu->yz[i][j][k]/grid->dz;
	        ptr->c = del*mu->yz[i][j][k+1]/grid->dz;
	        ptr->b = -ptr->a -ptr->c ;
            }
        } 
    }  
}  


void A_wmom(Grid *grid,Visc *mu,Velo *w)
{
    int i,j,k;
    double del;
    Tri_abc *ptr;
  
    for(i=w->istart;i<= w->iend;i++) {
        for(j=w->jstart;j<= w->jend;j++) {
            for(k=w->kstart;k<= w->kend;k++) {
                ptr    = &w->Ax[i][j][k];
                del    = 1.0/(Re*grid->dx);
	        ptr->a = del*mu->xz[i][j][k]/grid->dx;
	        ptr->c = del*mu->xz[i+1][j][k]/grid->dx;
	        ptr->b = -ptr->a -ptr->c ;

                ptr    = &w->Ay[i][j][k];
	        del    = 1.0/(Re*grid->dy);
	        ptr->a = del*mu->yz[i][j][k]/grid->dy;
	        ptr->c = del*mu->yz[i][j+1][k]/grid->dy;
	        ptr->b = -ptr->a -ptr->c ;

                ptr    = &w->Az[i][j][k];
                del    = 2.0/(Re*grid->dz);
	        ptr->a = del*mu->cell[i][j][k-1]/grid->dz;
	        ptr->c = del*mu->cell[i][j][k]/grid->dz;
	        ptr->b = -ptr->a -ptr->c ;
            }
        }
    }  
}


void Hc_op(Grid *grid,double ***un,double ***vn,double ***wn,Conc *c)
{
    int i,j,k,kk,sign;
    double *D1,*D2,***convec_x,***convec_y;
    double cx,cy,cz,vval,uval,wval,d2val;

    D1 = alloc1d_dble(grid->nmax_xyz); 
    D2 = alloc1d_dble(grid->nmax_xyz);
    convec_x = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);
    convec_y = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);

    for(j=c->jstart;j<= c->jend;j++) {
        for(k=c->kstart;k<= c->kend;k++) {

            for(i=0;i< c->nx_max-1; i++) {
	        D1[i] = (c->old[i+1][j][k] - c->old[i][j][k])/grid->dx;
	    }

	    for(i=c->istart;i<= c->iend; i++) {
	        D2[i] = (D1[i] - D1[i-1])/(2.0*grid->dx);
	    }

            for(i=c->istart;i<= c->iend; i++) {

	        uval = 0.5*(un[i+1][j][k]+un[i][j][k]);
	        sign = (uval>= 0.0) ? 1:-1 ;

	        if(((Rank == 0) & (i<= c->istart+1)) | ((Rank == Size-1) & (i>= c->iend-1))) {
	            kk = (sign ==1)? i-1 : i ;
	            d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
	            cx = D1[kk] + d2val*grid->dx*sign;
	        }
	        else {  
	     	    kk = i - 3 - (int)(2.5*(sign-1)); 
	            cx = calculate_weno(D1,kk,sign);
	        }
	        convec_x[i][j][k] = uval*cx;	     
            }
        }
    }

    for(i=c->istart;i<= c->iend;i++) {
        for(k=c->kstart;k<= c->kend;k++) {
	 
            for(j=0;j< c->ny_max-1; j++) {
	        D1[j] = (c->old[i][j+1][k] - c->old[i][j][k])/grid->dy; 
	    }
	    for(j=c->jstart;j<= c->jend; j++) {
	        D2[j] = (D1[j] - D1[j-1])/(2.0*grid->dy);
	    }
	       
            for(j=c->jstart;j<= c->jend; j++) {
	        vval = 0.5*(vn[i][j+1][k]+vn[i][j][k]);
	        sign = (vval>= 0.0) ? 1:-1 ;

	        if((j<= c->jstart+1) | (j>= c->jend-1)) {
	            kk = (sign ==1)? j-1 : j ; 
		    d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
		    cy = D1[kk] + d2val*grid->dy*sign;
	        }
	        else {  
	            kk = j - 3 - (int)(2.5*(sign-1)); 
	            cy = calculate_weno(D1,kk,sign);
	        }
	        convec_y[i][j][k] = vval*cy;
	    }   
        }
     }
   
     for(i=c->istart;i<= c->iend;i++) {
         for(j=c->jstart;j<= c->jend;j++) {

	     for(k=0;k< c->nz_max-1; k++) {
	         D1[k] = (c->old[i][j][k+1] - c->old[i][j][k])/grid->dz;
	     }

	     for(k=c->kstart;k<= c->kend; k++) {
	         D2[k] = (D1[k] - D1[k-1])/(2.0*grid->dz);
	     }
	       
             for(k=c->kstart;k<= c->kend; k++) {

	         wval = 0.5*(wn[i][j][k+1]+wn[i][j][k]);
	         sign = (wval>= 0.0) ? 1:-1 ;
	     
	         if((k<= c->kstart+1) | (k>= c->kend-1)) {
	             kk = (sign ==1)? k-1 : k ; 
		     d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
		     cz = D1[kk] + d2val*grid->dz*sign;
	         }
	         else {  
	             kk = k - 3 - (int)(2.5*(sign-1)); 
	             cz = calculate_weno(D1,kk,sign);
	         } 
	    
	         c->explicit_n[i][j][k] = -(wval*cz + convec_x[i][j][k] + convec_y[i][j][k]);
	     }   
         }
     }
 
    free(D1);
    free(D2);
    free3d_dble(grid->nx_max,grid->ny_max,grid->nz_max,convec_x);
    free3d_dble(grid->nx_max,grid->ny_max,grid->nz_max,convec_y);
     
}


void Hu_op(Grid *grid,Visc *mu,double ***vn,double ***wn,Velo *u)
{
    int i,j,k,kk,sign;
    double del,convec_term,vis_v1,vis_v2,vis_w1,vis_w2;
    double *D1,*D2,***convec_x,***convec_y;
    double ux,uy,uz,uval,vval,wval,d2val;
  
    D1 = alloc1d_dble(grid->nmax_xyz);
    D2 = alloc1d_dble(grid->nmax_xyz);
    convec_x = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);
    convec_y = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);
   
    for(j=u->jstart;j<= u->jend;j++) {
        for(k=u->kstart;k<= u->kend;k++) {

	    for(i=0;i< u->nx_max-1; i++) {
	        D1[i] = (u->old[i+1][j][k] - u->old[i][j][k])/grid->dx;
	    }
	    for(i=u->istart;i<= u->iend; i++) {
	        D2[i] = (D1[i] - D1[i-1])/(2.0*grid->dx);
	    }    
            for(i=u->istart;i<= u->iend; i++) {
	        uval = u->old[i][j][k];
	        sign = (uval>= 0.0) ? 1:-1 ;
	        if(((Rank == 0) &(i<= u->istart+1)) | ((Rank == Size-1) & (i>= u->iend-1))) {
	            kk = (sign ==1)? i-1 : i ;
		    d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
		    ux = D1[kk] + d2val*grid->dx*sign;
                }      
	        else {  
	            kk = i - 3 - (int)(2.5*(sign-1)); //starting pt. of divided diff
	            ux = calculate_weno(D1,kk,sign);
	        }
	        convec_x[i][j][k] = uval*ux;   
	    }
        }
    }
   

    for(i=u->istart;i<= u->iend;i++) {
        for(k=u->kstart;k<= u->kend;k++) {
          
	    for(j=0;j< u->ny_max-1; j++) {
	        D1[j] = (u->old[i][j+1][k] - u->old[i][j][k])/grid->dy;	 
	    }
	    for(j=u->jstart;j<= u->jend; j++) {
	        D2[j] = (D1[j] - D1[j-1])/(2.0*grid->dy);
	    }    
            for(j=u->jstart;j<= u->jend; j++) {
	        vval = 0.25*(vn[i-1][j][k]+vn[i][j][k]+vn[i][j+1][k]+vn[i-1][j+1][k]);
	        sign = (vval>= 0.0) ? 1:-1 ;
	        if((j<= u->jstart+1) | (j>= u->jend-1)) {
	            kk = (sign ==1)? j-1 : j ;
		    d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
		    uy = D1[kk] + d2val*grid->dy*sign;
	        }
	        else {  
	            kk = j - 3 - (int)(2.5*(sign-1));
	            uy = calculate_weno(D1,kk,sign);
	        }
	        convec_y[i][j][k] = vval*uy;
	    }   
        }
    }

    for(i=u->istart;i<= u->iend;i++) {
        for(j=u->jstart;j<= u->jend;j++) {

	    for(k=0;k< u->nz_max-1; k++) {
	        D1[k] = (u->old[i][j][k+1] - u->old[i][j][k])/grid->dz;	 
	    }
	    for(k=u->kstart;k<= u->kend; k++) {
	        D2[k] = (D1[k] - D1[k-1])/(2.0*grid->dz);
	    }
            for(k=u->kstart;k<= u->kend; k++) {
	        wval = 0.25*(wn[i-1][j][k]+wn[i][j][k]+wn[i][j][k+1]+wn[i-1][j][k+1]);
	        sign = (wval>= 0.0) ? 1:-1 ;
	        if((k<= u->kstart+1) | (k>= u->kend-1)) {
	            kk = (sign ==1)? k-1 : k ; 
		    d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
		    uz = D1[kk] + d2val*grid->dz*sign;
	        }
	        else {  
	            kk = k - 3 - (int)(2.5*(sign-1)); //starting pt. of divided diff
	            uz = calculate_weno(D1,kk,sign);
	        }

	        convec_term = wval*uz + convec_x[i][j][k] + convec_y[i][j][k];

                del   = 1.0/(Re*grid->dy*grid->dx);
	        vis_v1 = del*mu->xy[i][j+1][k]*(vn[i][j+1][k] - vn[i-1][j+1][k]);
	        vis_v2 = del*mu->xy[i][ j ][k]*(vn[i][ j ][k] - vn[i-1][ j ][k]);
	 
                del   = 1.0/(Re*grid->dz*grid->dx);
	        vis_w1 = del*mu->xz[i][j][k+1]*(wn[i][j][k+1] - wn[i-1][j][k+1]);
	        vis_w2 = del*mu->xz[i][j][ k ]*(wn[i][j][ k ] - wn[i-1][j][ k ]);

                u->explicit_n[i][j][k] = -convec_term + vis_v1 - vis_v2 + vis_w1 - vis_w2;
	    }   
        }
    }
 
    free(D1);
    free(D2);
    free3d_dble(grid->nx_max,grid->ny_max,grid->nz_max,convec_x);
    free3d_dble(grid->nx_max,grid->ny_max,grid->nz_max,convec_y);
    
}


void Hv_op(Grid *grid,Visc *mu,double ***un,double ***wn,Velo *v)
{
    int i,j,k,kk,sign;
    double del,convec_term,vis_u1,vis_u2,vis_w1,vis_w2;
    double *D1,*D2,***convec_x,***convec_y;
    double vx,vy,vz,vval,uval,wval,d2val;

    D1 = alloc1d_dble(grid->nmax_xyz); 
    D2 = alloc1d_dble(grid->nmax_xyz);
    convec_x = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);
    convec_y = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);
  
   
    for(j=v->jstart;j<= v->jend;j++) {
        for(k=v->kstart;k<= v->kend;k++) {

            for(i=0;i< v->nx_max-1; i++) {
	        D1[i] = (v->old[i+1][j][k] - v->old[i][j][k])/grid->dx;
	    }
	    for(i=v->istart;i<= v->iend; i++) {
	        D2[i] = (D1[i] - D1[i-1])/(2.0*grid->dx);
	    }
	    for(i=v->istart;i<= v->iend; i++) {
	        uval = 0.25*(un[i][j-1][k]+un[i+1][j-1][k]+un[i][j][k]+un[i+1][j][k]);
	        sign = (uval>= 0.0) ? 1:-1 ;
	        if( ((Rank == 0) & (i<= v->istart+1)) | ((Rank == Size-1) & (i>= v->iend-1)) ){
	            kk = (sign ==1)? i-1 : i ;
	            d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
	            vx = D1[kk] + d2val*grid->dx*sign;	
	        }
	        else {  
	            kk = i - 3 - (int)(2.5*(sign-1));
	            vx = calculate_weno(D1,kk,sign);
	        }
	        convec_x[i][j][k] = uval*vx;
            }   
        }
    }

  
    for(i=v->istart;i<= v->iend;i++) {
        for(k=v->kstart;k<= v->kend;k++) {
        
	    for(j=0;j< v->ny_max-1; j++) {
	        D1[j] = (v->old[i][j+1][k] - v->old[i][j][k])/grid->dy;  	 
	    }
	    for(j=v->jstart;j<= v->jend; j++) {
	        D2[j] = (D1[j] - D1[j-1])/(2.0*grid->dy);
	    }
	    for(j=v->jstart;j<= v->jend; j++) {
	        vval = v->old[i][j][k];
	        sign = (vval>= 0.0) ? 1:-1 ;
                if((j<= v->jstart+1) | (j>= v->jend-1)) {
	            kk = (sign ==1)? j-1 : j ;
	            d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
	            vy = D1[kk] + d2val*grid->dy*sign;
                }
	        else {  
	            kk = j - 3 - (int)(2.5*(sign-1));
	            vy = calculate_weno(D1,kk,sign);
                }
	        convec_y[i][j][k] = vval*vy;
            }   
        }
    }


    for(i=v->istart;i<= v->iend;i++) {
        for(j=v->jstart;j<= v->jend;j++) {

            for(k=0;k< v->nz_max-1; k++) {
	        D1[k] = (v->old[i][j][k+1] - v->old[i][j][k])/grid->dz;
	    }
            for(k=v->kstart;k<= v->kend; k++) {
	        D2[k] = (D1[k] - D1[k-1])/(2.0*grid->dz);
	    }   
            for(k=v->kstart;k<= v->kend; k++) {
	        wval = 0.25*(wn[i][j-1][k]+wn[i][j-1][k+1]+wn[i][j][k]+wn[i][j][k+1]);
	        sign = (wval>= 0.0) ? 1:-1 ;
	        if((k<= v->kstart+1) | (k>= v->kend-1)) {
	            kk = (sign ==1)? k-1 : k ;
	            d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
	            vz = D1[kk] + d2val*grid->dz*sign;
                }
	        else {  
                    kk = k - 3 - (int)(2.5*(sign-1));
	            vz = calculate_weno(D1,kk,sign);
	        }

	        convec_term = wval*vz + convec_x[i][j][k] + convec_y[i][j][k];

                del  = 1.0/(Re*grid->dx*grid->dy);
                vis_u1 = del*mu->xy[i+1][j][k]*(un[i+1][j][k]-un[i+1][j-1][k]);
	        vis_u2 = del*mu->xy[i][j][k]*(un[i][j][k]-un[i][j-1][k]);
	   
                del  = 1.0/(Re*grid->dz*grid->dy);
                vis_w1 = del*mu->yz[i][j][k+1]*(wn[i][j][k+1]-wn[i][j-1][k+1]);
	        vis_w2 = del*mu->yz[i][j][k]*(wn[i][j][k]-wn[i][j-1][k]);
	   
                v->explicit_n[i][j][k] = -convec_term+vis_u1-vis_u2+vis_w1-vis_w2;
            }   
        }
    }
    
    free(D1);
    free(D2);
    free3d_dble(grid->nx_max,grid->ny_max,grid->nz_max,convec_x);
    free3d_dble(grid->nx_max,grid->ny_max,grid->nz_max,convec_y);
   
}


void Hw_op(Grid *grid,Visc *mu,double ***un,double ***vn,Velo *w)
{
    int i,j,k,kk,sign;
    double del,convec_term,vis_u1,vis_u2,vis_v1,vis_v2;
    double *D1,*D2,***convec_x,***convec_y;
    double wx,wy,wz,vval,uval,wval,d2val;

    D1 = alloc1d_dble(grid->nmax_xyz);
    D2 = alloc1d_dble(grid->nmax_xyz);
    convec_x = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);
    convec_y = alloc3d_dble(grid->nx_max,grid->ny_max,grid->nz_max);
  
    for(j=w->jstart;j<= w->jend;j++) {
        for(k=w->kstart;k<= w->kend;k++) {

	    for(i=0;i< w->nx_max-1; i++) {
	        D1[i] = (w->old[i+1][j][k] - w->old[i][j][k])/grid->dx;
            }

            for(i=w->istart;i<= w->iend; i++) {
                D2[i] = (D1[i] - D1[i-1])/(2.0*grid->dx);
	    }
            for(i=w->istart;i<= w->iend; i++) {
                uval = 0.25*(un[i][j][k-1]+un[i+1][j][k-1]+un[i][j][k]+un[i+1][j][k]);
	        sign = (uval>= 0.0) ? 1:-1 ;
	        if(((Rank == 0) & (i<= w->istart+1)) | ((Rank == Size-1) & (i>= w->iend-1))) {
	            kk = (sign ==1)? i-1 : i ;
	            d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
	            wx = D1[kk] + d2val*grid->dx*sign;
	        }
	        else {  
	            kk = i - 3 - (int)(2.5*(sign-1)); 
	            wx = calculate_weno(D1,kk,sign);
	        }
	        convec_x[i][j][k] = uval*wx;
	    }   
        }
    }
   

    for(i=w->istart;i<= w->iend;i++) {
        for(k=w->kstart;k<= w->kend;k++) {
 
	    for(j=0;j< w->ny_max-1; j++) {
	        D1[j] = (w->old[i][j+1][k] - w->old[i][j][k])/grid->dy; 
	    }
	    for(j=w->jstart;j<= w->jend; j++) {
	        D2[j] = (D1[j] - D1[j-1])/(2.0*grid->dy);
	    }    
            for(j=w->jstart;j<= w->jend; j++) {
	        vval = 0.25*(vn[i][j][k-1]+vn[i][j][k]+vn[i][j+1][k-1]+vn[i][j+1][k]);
	        sign = (vval>= 0.0) ? 1:-1 ;
	        if((j<= w->jstart+1) | (j>= w->jend-1)) {
	            kk = (sign ==1)? j-1 : j ; 
	            d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
	            wy = D1[kk] + d2val*grid->dy*sign;
	        }
	        else {  
	            kk = j - 3 - (int)(2.5*(sign-1));
                    wy = calculate_weno(D1,kk,sign);
	        }
	        convec_y[i][j][k] = vval*wy;
            }   
        }
    }

  
    for(i=w->istart;i<= w->iend;i++) {
        for(j=w->jstart;j<= w->jend;j++) {

	    for(k=0;k< w->nz_max-1; k++) {
	        D1[k] = (w->old[i][j][k+1] - w->old[i][j][k])/grid->dz;
	    }
            for(k=w->kstart;k<= w->kend; k++) {
	        D2[k] = (D1[k] - D1[k-1])/(2.0*grid->dz);
	    } 
            for(k=w->kstart;k<= w->kend; k++) {
	        wval = w->old[i][j][k];
                sign = (wval>= 0.0) ? 1:-1 ;
	        if((k<= w->kstart+1) | (k>= w->kend-1)) {
	            kk = (sign ==1)? k-1 : k ;
	            d2val = (fabs(D2[kk]) <= fabs(D2[kk+1])) ? D2[kk] : D2[kk+1];
	            wz = D1[kk] + d2val*grid->dz*sign;
	        }
	        else {  
	            kk = k - 3 - (int)(2.5*(sign-1)); 
	            wz = calculate_weno(D1,kk,sign);
	        }
	     
	        convec_term = wval*wz + convec_x[i][j][k] + convec_y[i][j][k];
	   
                del    = 1.0/(Re*grid->dx*grid->dz);
                //vis_u1 = del*mu->xz[i][j][k+1]*(un[i+1][j][k]-un[i+1][j][k-1]);
		vis_u1 = del*mu->xz[i+1][j][k]*(un[i+1][j][k]-un[i+1][j][k-1]);
		
	        vis_u2 = del*mu->xz[i][j][k]*(un[i][j][k]-un[i][j][k-1]);

                del    = 1.0/(Re*grid->dy*grid->dz);
                vis_v1 = del*mu->yz[i][j+1][k]*(vn[i][j+1][k]-vn[i][j+1][k-1]);
	        vis_v2 = del*mu->yz[i][j][k]*(vn[i][j][k]-vn[i][j][k-1]);

                w->explicit_n[i][j][k] = -convec_term+vis_u1-vis_u2+vis_v1-vis_v2;
	    }  
        }
    }

    free(D1);
    free(D2);
    free3d_dble(grid->nx_max,grid->ny_max,grid->nz_max,convec_x);
    free3d_dble(grid->nx_max,grid->ny_max,grid->nz_max,convec_y);
  
}


void divergence(Grid *grid,double ***un,double ***vn,double ***wn,double ***div)
{
    int i,j,k;
    double divx,divy,divz;
 
    for(i=grid->istart;i<= grid->iend;i++) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
	        divx = (un[i+1][j][k] - un[i][j][k])/grid->dx;
                divy = (vn[i][j+1][k] - vn[i][j][k])/grid->dy;
                divz = (wn[i][j][k+1] - wn[i][j][k])/grid->dz;
                div[i][j][k] = (divx + divy + divz);
            }
        }
    }
    		
}


void phisolver_dct(double dt,double rk_gamma,double rk_rho,Grid *grid,double ***div,double ***phi)
{
    int i,j,k,nx,ny,nz,nsys,tri_index;
    double phiconst,ki,kk,rk_alpha,fac,dx2i;
    double ***source;
    Tds *ptr,*tri;

    rk_alpha = rk_gamma + rk_rho;
    fac = Re/(dt*rk_alpha);
  
    nx = grid->nx_cell;
    ny = grid->ny_cell;
    nz = grid->nz_cell;

    source = alloc3d_dble(ny,nz,nx);
   
    for(i=grid->istart;i<= grid->iend;i++) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
                source[j-grid->jstart][k-grid->kstart][i-grid->istart] = fac*div[i][j][k];
            }
        }   	 
    }
    /*
    for(i=grid->istart;i<= grid->iend;i++) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
                //source[j-grid->jstart][k-grid->kstart][i-grid->istart] = -8*PI*PI*cos(2*PI*grid->Yc[j])*cos(2*PI*grid->Zc[k]);
            }
        }   	 
    }
    */ 

    fdct_3d(ny,nz,nx,source,source);

    nsys = ny*nz;
    tri = alloc_tds(nsys,nx);    
  
    phiconst = 0.0;
    dx2i = 1.0/(grid->dx*grid->dx);
  
    tri_index = 0;
    for(j=0;j< ny;j++) {
        for(k=0;k< nz;k++) {

            ptr = tri + tri_index;
	    tri_index++;

            ki = 2.0*(cos(PI*(double)j/(double)ny)-1.0)/(grid->dy*grid->dy);
            kk = 2.0*(cos(PI*(double)k/(double)nz)-1.0)/(grid->dz*grid->dz);

            for(i=0;i< nx;i++) {
                ptr->acof[i] = dx2i;
  	        ptr->bcof[i] = -2.0*dx2i + ki + kk;
	        ptr->ccof[i] = dx2i;
	        ptr->rhs[i]  = source[j][k][i];
            }
	    // left boundary: Neumann 
            if(Rank == 0) {
                ptr->bcof[0] += ptr->acof[0];
            }
            if(Rank == Size-1) {  
                
		/*first mode singular;solve with a specified constant */
                /*ghost cell phi set to a const;*/
		if((j ==0)&&(k==0)) {
                    ptr->rhs[nx-1] -= ptr->ccof[nx-1]*phiconst;
                }
                else {
	            ptr->bcof[nx-1] += ptr->ccof[nx-1]; // right boundary: Neumann
                }
            }	  
        }
    }
    
    /*
    tri_index = 0;
    for(j=0;j< ny;j++) {
        for(k=0;k< nz;k++) {
            ptr = tri + tri_index;
            tri_index++;
            for(i=0;i< nx;i++) {
                //printf("%f %f %f\n",ptr->acof[i],ptr->bcof[i],ptr->ccof[i]);
            }  
        }       
    }
    //getchar();
    */

    pll_tridiag(tri,nsys,nx);
   
    tri_index = 0;
    for(j=0;j< ny;j++) {
        for(k=0;k< nz;k++) {
            ptr = tri + tri_index;
            tri_index++;
            for(i=0;i< nx;i++) {
                source[j][k][i] = ptr->rhs[i];
            }  
        }       
    }

    idct_3d(ny,nz,nx,source,source);

    for(i=grid->istart;i<= grid->iend;i++) {
        for(j=grid->jstart;j<= grid->jend;j++) {
           for(k=grid->kstart;k<= grid->kend;k++) {
	       phi[i][j][k] = source[j-grid->jstart][k-grid->kstart][i-grid->istart];
           }
        }
    }
    
    /*
    //if(Rank == 0){
        //for(i=grid->istart;i<= grid->iend;i++) {
            for(j=grid->jstart;j<= grid->jend;j++) {
                for(k=grid->kstart;k<= grid->kend;k++) {
                     //printf("phi[%d][%d][%d] = %f\n",3,j,k,phi[3][j][k]);
                }
	    }
	//}
    }
    //getchar(); 
    */
    
    free_tds(tri,nsys,nx);
    free3d_dble(ny,nz,nx,source);
  
}


void velo_correction(double dt,double rk_gamma,double rk_rho,Grid *grid, double ***phi,Velo *u,Velo *v,Velo *w)
{
    int i,j,k,ierr;
    double rk_alpha,fac,del_phi,deficit,qin,qout;
    double *q_k;
    MPI_Status status; 
    MPI_Request request;

    rk_alpha = rk_gamma + rk_rho;
    fac = dt*rk_alpha/Re;

    for(i=v->istart;i<= v->iend;i++) {
        for(j=v->jstart;j<= v->jend;j++) {
            for(k=v->kstart;k<= v->kend; k++) {
	        del_phi = (phi[i][j][k] - phi[i][j-1][k])/grid->dy;
	        v->new[i][j][k] = v->new[i][j][k] - fac*del_phi;
            }
        }
    }  

    for(i=w->istart;i<= w->iend;i++) {
        for(j=w->jstart;j<= w->jend;j++) {
            for(k=w->kstart;k<= w->kend; k++) {
	        del_phi = (phi[i][j][k] - phi[i][j][k-1])/grid->dz;
	        w->new[i][j][k] = w->new[i][j][k] - fac*del_phi;
            }
        }
    } 

    for(i=u->istart;i<= u->iend;i++) {
        for(j=u->jstart;j<= u->jend;j++) {
            for(k=u->kstart;k<= u->kend; k++) {
	        del_phi = (phi[i][j][k]- phi[i-1][j][k])/grid->dx;
                u->new[i][j][k] = u->new[i][j][k] - fac*del_phi;
            }
        }
    }

    if(Rank == 0 ) {
        // inflow flow rate
        q_k  = alloc1d_dble(u->ny_max);
        i = u->istart - 1;
        for(j=u->jstart;j<= u->jend;j++) { //two-dimensional integration 
            qin = 0.0;
	    for(k=u->kstart+1;k<= u->kend;k++) { //integrate along k-plane
	        qin  += 0.5*grid->dz*(u->new[i][j][k] + u->new[i][j][k-1]);
	    }
            q_k[j]  = qin;
        } 
        qin = 0.0;
        for(j=u->jstart+1;j<= u->jend;j++) { //integrate along j-plane
            qin  += 0.5*grid->dy*(q_k[j-1] + q_k[j]);
        } 
	
      
        ierr = MPI_Isend(&qin,1, MPI_DOUBLE,Size-1,1,MPI_COMM_WORLD,&request);
        free(q_k);
   
    }
    if(Rank == Size-1) {
        
        // Initiate the non-blocking receive from root to get inflow flow rate
        ierr = MPI_Irecv(&qin,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&request);

        // outflow flow rate
        q_k  = alloc1d_dble(u->ny_max);
        i = u->iend + 1;
        for(j=u->jstart;j<= u->jend;j++) { //two-dimensional integration 
   	    qout = 0.0;
   	    for(k=u->kstart+1;k<= u->kend;k++) { //integrate along k-plane
   	        qout  += 0.5*grid->dz*(u->new[i][j][k] + u->new[i][j][k-1]);
   	    }
   	    q_k[j]  = qout;
        } 
        qout = 0.0;
        for(j=u->jstart+1;j<= u->jend;j++) { //integrate along j-plane
   	    qout  += 0.5*grid->dy*(q_k[j-1] + q_k[j]);
        } 

        //wait to get the inflow flow rate
        ierr =  MPI_Wait(&request,&status);

        /*average velocity deficit */
        deficit = (qin - qout)/((Y2-Y1)*(Z2-Z1));
  
        /* add deficit to the vz */
        for(j=u->jstart;j<= u->jend;j++) {
   	    for(k=u->kstart;k<= u->kend; k++) {
   	        u->new[u->iend+1][j][k] = u->new[u->iend+1][j][k] + deficit;
   	    }   
        }
        free(q_k);
    } 
     
}


void exchange_conc_boundary(Conc *c)
{
    int i,j,k,index,ileft,iright,ierr;
    int ntotal,nghost_cells,left_proc,right_proc,left_mesg_tag,right_mesg_tag;
    double *left_send,*left_recv,*right_send,*right_recv;
    MPI_Status status; 
    MPI_Request request[4];
   
    nghost_cells = c->istart; // number of ghost cells
    ntotal =  c->ny_max*c->nz_max*nghost_cells;
   
    left_send  = (double*)malloc(sizeof(double)*ntotal);
    left_recv  = (double*)malloc(sizeof(double)*ntotal);
    right_send = (double*)malloc(sizeof(double)*ntotal);
    right_recv = (double*)malloc(sizeof(double)*ntotal);
   
    left_proc  = (Rank == 0)     ? MPI_PROC_NULL : Rank - 1;
    right_proc = (Rank == Size-1)? MPI_PROC_NULL : Rank + 1;
    left_mesg_tag  = 1;
    right_mesg_tag = 2;
   
    ierr = MPI_Irecv(left_recv, ntotal,MPI_DOUBLE,left_proc,right_mesg_tag,MPI_COMM_WORLD,(request+0));
    ierr = MPI_Irecv(right_recv,ntotal,MPI_DOUBLE,right_proc,left_mesg_tag,MPI_COMM_WORLD,(request+1));

    index = 0;
    for(i=0;i< nghost_cells; i++) { 
        ileft  = c->istart + i;
        //iright = c->iend - i;
        iright = (c->iend-nghost_cells+1)+ i;
        for(j=0;j< c->ny_max;j++) {
            for(k=0;k< c->nz_max;k++) {
	        left_send[index]    = c->new[ileft][j][k];
	        right_send[index]   = c->new[iright][j][k];
	        index += 1;
	    }
        }
    }
   
    ierr = MPI_Isend(left_send,ntotal, MPI_DOUBLE,left_proc,  left_mesg_tag,MPI_COMM_WORLD,(request+2));
    ierr = MPI_Isend(right_send,ntotal,MPI_DOUBLE,right_proc,right_mesg_tag,MPI_COMM_WORLD,(request+3));

    if(Rank != 0) {
        ierr = MPI_Wait(request,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { 
            ileft  = i;
            for(j=0;j< c->ny_max;j++) {
                for(k=0;k< c->nz_max;k++) {
	            c->new[ileft][j][k] = left_recv[index];
	            index += 1;
	        }
            }
        } 
    }

    if(Rank != Size-1) {
        ierr = MPI_Wait(request+1,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { 
            //iright = c->iend + i;
	    iright = (c->iend+1) + i;
            for(j=0;j< c->ny_max;j++) {
                for(k=0;k< c->nz_max;k++) {
	            c->new[iright][j][k] = right_recv[index];
	            index += 1;
	        }
            }
        } 
    } 

    for(i=2;i<=3;i++) { 
        MPI_Wait(request+i,&status);
    } 
      
    free(left_send);
    free(left_recv);
    free(right_send);
    free(right_recv);

}


void exchange_u(Grid *grid,Velo *u)
{
    int i,j,k,index,ileft,iright,ierr;
    int ntotal,nghost_cells,left_proc,right_proc,left_mesg_tag,right_mesg_tag;
    double *left_send,*left_recv,*right_send,*right_recv;
    MPI_Status status; 
    MPI_Request request[4];
   
    nghost_cells = 1; // number of ghost cells
    ntotal =  grid->ny_max*grid->nz_max*nghost_cells*1; // only u variable
   
    //allocate the buffer arrays
    left_send  = (double*)malloc(sizeof(double)*ntotal);
    left_recv  = (double*)malloc(sizeof(double)*ntotal);
    right_send = (double*)malloc(sizeof(double)*ntotal);
    right_recv = (double*)malloc(sizeof(double)*ntotal);
   
    left_proc  = (Rank == 0)? MPI_PROC_NULL : Rank - 1;
    right_proc = (Rank == Size-1)? MPI_PROC_NULL : Rank + 1;
    left_mesg_tag  = 1;
    right_mesg_tag = 2;
   
    ierr = MPI_Irecv(left_recv,ntotal, MPI_DOUBLE,left_proc,right_mesg_tag,MPI_COMM_WORLD,(request+0));
    ierr = MPI_Irecv(right_recv,ntotal,MPI_DOUBLE,right_proc,left_mesg_tag,MPI_COMM_WORLD,(request+1));

    index = 0;
    for(i=0;i< nghost_cells; i++) { 
        ileft  = grid->istart + i;
        //iright = grid->iend - i;
        iright = (grid->iend-nghost_cells+1)+ i;
        for(j=0;j< grid->ny_max;j++) {
            for(k=0;k< grid->nz_max;k++) {
	        left_send[index]  = u->new[ileft][j][k];
	        right_send[index] = u->new[iright][j][k];
	        index += 1;
	    }
        }
    }
   
    /* left_send  goes to right_recv */
    /* right_send goes to left_recv */
   
    ierr = MPI_Isend(left_send,ntotal, MPI_DOUBLE,left_proc,  left_mesg_tag,MPI_COMM_WORLD,(request+2));
    ierr = MPI_Isend(right_send,ntotal,MPI_DOUBLE,right_proc,right_mesg_tag,MPI_COMM_WORLD,(request+3));

    if(Rank != 0) {
        ierr =  MPI_Wait(request,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) {
            ileft  = (grid->istart -1) +i;
            for(j=0;j< grid->ny_max;j++) {
                for(k=0;k< grid->nz_max;k++) {
	            u->new[ileft][j][k] = left_recv[index];
	            index += 1;
	        }
            }
        } 
    }
 
    if(Rank != Size-1) {
        ierr =  MPI_Wait(request+1,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { 
            //iright = grid->iend + i;
	    iright = (grid->iend+1) + i;
            for(j=0;j< grid->ny_max;j++) {
                for(k=0;k< grid->nz_max;k++) {
	            u->new[iright][j][k] = right_recv[index];
	    	    index += 1;
	        }
            }
        } 
    } 

    for(i=2;i<=3;i++) { 
        MPI_Wait(request+i,&status);
    }   
   
    free(left_send);
    free(left_recv);
    free(right_send);
    free(right_recv);

}


void exchange_phi(Grid *grid,double ***phi)
{
    int i,j,k,index,ileft,iright,ierr;
    int ntotal,nghost_cells,left_proc,right_proc,left_mesg_tag,right_mesg_tag;
    double *left_send,*left_recv,*right_send,*right_recv;
    MPI_Status status; 
    MPI_Request request[4];
   
    nghost_cells = grid->istart; 
    ntotal =  grid->ny_max*grid->nz_max*nghost_cells*1;

    left_send  = (double*)malloc(sizeof(double)*ntotal);
    left_recv  = (double*)malloc(sizeof(double)*ntotal);
    right_send = (double*)malloc(sizeof(double)*ntotal);
    right_recv = (double*)malloc(sizeof(double)*ntotal);
   
    left_proc  = (Rank == 0)? MPI_PROC_NULL : Rank - 1;
    right_proc = (Rank == Size-1)? MPI_PROC_NULL : Rank + 1;
    left_mesg_tag  = 1;
    right_mesg_tag = 2;
   
    ierr = MPI_Irecv(left_recv, ntotal,MPI_DOUBLE,left_proc,right_mesg_tag,MPI_COMM_WORLD,(request+0));
    ierr = MPI_Irecv(right_recv,ntotal,MPI_DOUBLE,right_proc,left_mesg_tag,MPI_COMM_WORLD,(request+1));

    index = 0;
    for(i=0;i< nghost_cells; i++) { 
        ileft  = grid->istart + i;
        //iright = grid->iend - i;
        iright = (grid->iend-nghost_cells+1)+ i;
        for(j=0;j< grid->ny_max;j++) {
            for(k=0;k< grid->nz_max;k++) {
	        left_send[index] = phi[ileft][j][k];
	        right_send[index] = phi[iright][j][k];
	        index += 1;
	    }
        }
    }

    ierr = MPI_Isend(left_send, ntotal,MPI_DOUBLE,left_proc,  left_mesg_tag,MPI_COMM_WORLD,(request+2));
    ierr = MPI_Isend(right_send,ntotal,MPI_DOUBLE,right_proc,right_mesg_tag,MPI_COMM_WORLD,(request+3));

    if(Rank != 0) {  
        ierr =  MPI_Wait(request,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { 
            ileft  = i;
            for(j=0;j< grid->ny_max;j++) {
                for(k=0;k< grid->nz_max;k++) {
	            phi[ileft][j][k]    = left_recv[index];
	            index += 1;
	        }
            }
        } 
    }

    if(Rank != Size-1) {  
        ierr =  MPI_Wait(request+1,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { //3 points
            //iright = grid->iend + i;
	    iright = (grid->iend+1) + i;
            for(j=0;j< grid->ny_max;j++) {
                for(k=0;k< grid->nz_max;k++) {
	            phi[iright][j][k]    = right_recv[index];
	            index += 1;
	        }
            }
        } 
    } 
    
    //wait to finish the send operation before freeing
    for(i=2;i<=3;i++) { 
        MPI_Wait(request+i,&status);
    }   
   
    free(left_send);
    free(left_recv);
    free(right_send);
    free(right_recv);
 
}


void exchange_velo_boundary(Grid *grid,Velo *u,Velo *v,Velo *w)
{
    int i,j,k,index,ileft,iright,ierr;
    int ntotal,nghost_cells,left_proc,right_proc,left_mesg_tag,right_mesg_tag;
    double *left_send,*left_recv,*right_send,*right_recv;
    MPI_Status status; 
    MPI_Request request[4];
   
    nghost_cells = grid->istart; // number of ghost cells
    /* total number of ghost points to send */
    ntotal =  grid->ny_max*grid->nz_max*nghost_cells*3; // 3 velocity variables
   
    //allocate the buffer arrays
    left_send  = (double*)malloc(sizeof(double)*ntotal);
    left_recv  = (double*)malloc(sizeof(double)*ntotal);
    right_send = (double*)malloc(sizeof(double)*ntotal);
    right_recv = (double*)malloc(sizeof(double)*ntotal);
   
    left_proc  = (Rank == 0)?      MPI_PROC_NULL : Rank - 1;
    right_proc = (Rank == Size-1)? MPI_PROC_NULL : Rank + 1;
    left_mesg_tag  = 1;
    right_mesg_tag = 2;
   
    /* Initiate the non-blocking receive: note the change in mesg_tag */
    ierr = MPI_Irecv(left_recv,ntotal, MPI_DOUBLE,left_proc,right_mesg_tag,MPI_COMM_WORLD,(request+0));
    ierr = MPI_Irecv(right_recv,ntotal,MPI_DOUBLE,right_proc,left_mesg_tag,MPI_COMM_WORLD,(request+1));
   
    /*copy the left & right 3 points to to be send to left&right proc*/
    index = 0;
    for(i=0;i< nghost_cells; i++) { 
        ileft  = grid->istart + i;
        //iright = grid->iend - i;
        iright = (grid->iend-nghost_cells+1)+ i;
        for(j=0;j< grid->ny_max;j++) {
            for(k=0;k< grid->nz_max;k++) {
	        // left most points
	        left_send[index]   = u->new[ileft][j][k];
	        left_send[index+1] = v->new[ileft][j][k];
	        left_send[index+2] = w->new[ileft][j][k];
	        /* left_send[index+3] = phi[ileft][j][k]; */
	        //right most points
	        right_send[index]   = u->new[iright][j][k];
	        right_send[index+1] = v->new[iright][j][k];
	        right_send[index+2] = w->new[iright][j][k];
	        /* right_send[index+3] = phi[iright][j][k]; */
	      
	        index += 3;
	    }
        }
    }
   
    /* Initiate the non-blocking send */
    ierr = MPI_Isend(left_send, ntotal,MPI_DOUBLE,left_proc,  left_mesg_tag,MPI_COMM_WORLD,(request+2));
    ierr = MPI_Isend(right_send,ntotal,MPI_DOUBLE,right_proc,right_mesg_tag,MPI_COMM_WORLD,(request+3));
   
    /*update the ghost cells received from left proc*/ 
    if(Rank != 0) {
        //complete the receive communication before updating boundary values  
        ierr =  MPI_Wait(request,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { //3 points
            ileft  = i;
            for(j=0;j< grid->ny_max;j++) {
                for(k=0;k< grid->nz_max;k++) {
	            u->new[ileft][j][k] = left_recv[index];
	            v->new[ileft][j][k] = left_recv[index+1];
	            w->new[ileft][j][k] = left_recv[index+2];
	            /* phi[ileft][j][k]    = left_recv[index+3]; */
	     
	            index += 3;
	        }
            }
        }
    }
    
    /*update the ghost cells received from right proc*/ 
    if(Rank != Size-1) {
        //complete the receive communication before updating boundary values  
        ierr =  MPI_Wait(request+1,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { //3 points
            //iright = grid->iend + i;
	    iright = (grid->iend+1) + i;
            for(j=0;j< grid->ny_max;j++) {
                for(k=0;k< grid->nz_max;k++) {
	            u->new[iright][j][k] = right_recv[index];
	            v->new[iright][j][k] = right_recv[index+1];
	            w->new[iright][j][k] = right_recv[index+2];
	            /* phi[iright][j][k]    = right_recv[index+3]; */
	     
	            index += 3;
	        }
            }
        } 
    } 
  
    //wait to finish the send operation before freeing
    for(i=2;i<=3;i++) { 
        MPI_Wait(request+i,&status);
    }   
   
    free(left_send);
    free(left_recv);
    free(right_send);
    free(right_recv);

}


void exchange_velophi_boundary(Grid *grid,double ***phi,Velo *u,Velo *v,Velo *w)
{
    int i,j,k,index,ileft,iright,ierr;
    int ntotal,nghost_cells,left_proc,right_proc,left_mesg_tag,right_mesg_tag;
    double *left_send,*left_recv,*right_send,*right_recv;
    MPI_Status status; 
    MPI_Request request[4];
   
    nghost_cells = grid->istart; // number of ghost cells
    /* total number of ghost points to send */
    ntotal =  grid->ny_max*grid->nz_max*nghost_cells*4; // All 4 variables
   
    //allocate the buffer arrays
    left_send  = (double*)malloc(sizeof(double)*ntotal);
    left_recv  = (double*)malloc(sizeof(double)*ntotal);
    right_send = (double*)malloc(sizeof(double)*ntotal);
    right_recv = (double*)malloc(sizeof(double)*ntotal);
   
    left_proc  = (Rank == 0)?      MPI_PROC_NULL : Rank - 1;
    right_proc = (Rank == Size-1)? MPI_PROC_NULL : Rank + 1;
    left_mesg_tag  = 1;
    right_mesg_tag = 2;
   
    /* Initiate the non-blocking receive: note the change in mesg_tag */
    ierr = MPI_Irecv(left_recv,ntotal, MPI_DOUBLE,left_proc,right_mesg_tag,MPI_COMM_WORLD,(request+0));
    ierr = MPI_Irecv(right_recv,ntotal,MPI_DOUBLE,right_proc,left_mesg_tag,MPI_COMM_WORLD,(request+1));
   
    /*copy the left & right 3 points to to be send to left&right proc*/
    index = 0;
    for(i=0;i< nghost_cells; i++) { 
        ileft  = grid->istart + i;
        //iright = grid->iend - i;
        iright = (grid->iend-nghost_cells+1)+ i;
        for(j=0;j< grid->ny_max;j++) {
            for(k=0;k< grid->nz_max;k++) {
	        // left most points
	        left_send[index]   = u->new[ileft][j][k];
	        left_send[index+1] = v->new[ileft][j][k];
	        left_send[index+2] = w->new[ileft][j][k];
	        left_send[index+3] = phi[ileft][j][k];
	        //right most points
	        right_send[index]   = u->new[iright][j][k];
	        right_send[index+1] = v->new[iright][j][k];
	        right_send[index+2] = w->new[iright][j][k];
	        right_send[index+3] = phi[iright][j][k];
	      
	        index += 4;
	    }
        }
    }
   
    /* Initiate the non-blocking send */
    ierr = MPI_Isend(left_send, ntotal,MPI_DOUBLE,left_proc,  left_mesg_tag,MPI_COMM_WORLD,(request+2));
    ierr = MPI_Isend(right_send,ntotal,MPI_DOUBLE,right_proc,right_mesg_tag,MPI_COMM_WORLD,(request+3));
   
    /*update the ghost cells received from left proc*/ 
    if(Rank != 0) {
        //complete the receive communication before updating boundary values  
        ierr =  MPI_Wait(request,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { //3 points
            ileft  = i;
            for(j=0;j< grid->ny_max;j++) {
                for(k=0;k< grid->nz_max;k++) {
	            u->new[ileft][j][k] = left_recv[index];
	            v->new[ileft][j][k] = left_recv[index+1];
	            w->new[ileft][j][k] = left_recv[index+2];
	            phi[ileft][j][k]    = left_recv[index+3];
	     
	            index += 4;
	        }
            }
        }
    }
    
    /*update the ghost cells received from right proc*/ 
    if(Rank != Size-1) {
        //complete the receive communication before updating boundary values  
        ierr =  MPI_Wait(request+1,&status);
        index = 0;
        for(i=0;i< nghost_cells; i++) { //3 points
            //iright = grid->iend + i;
	    iright = (grid->iend+1) + i;
            for(j=0;j< grid->ny_max;j++) {
                for(k=0;k< grid->nz_max;k++) {
	            u->new[iright][j][k] = right_recv[index];
	            v->new[iright][j][k] = right_recv[index+1];
	            w->new[iright][j][k] = right_recv[index+2];
	            phi[iright][j][k]    = right_recv[index+3];
	     
	            index += 4;
	        }
            }
        } 
    } 
  
    //wait to finish the send operation before freeing
    for(i=2;i<=3;i++) { 
        MPI_Wait(request+i,&status);
    }   
   
    free(left_send);
    free(left_recv);
    free(right_send);
    free(right_recv);

}



/* update first u-boundary node to compute divergence */
void update_velo_boundary(Grid *grid,Velo *u,Velo *v,Velo *w)
{
    int i,j,k,in,out;
    void exchange_u(Grid *,Velo*);

    /* update ghost point at side face */
    /* note you donot need to update w-velo, for it is zero at the face */
    for(i=grid->istart;i<= grid->iend;i++) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            //front side : symmetry
	    in  = grid->kstart;
	    out = grid->kstart-1;
	    u->new[i][j][out] = u->new[i][j][in]; //neumann
	    v->new[i][j][out] = v->new[i][j][in]; //neumann
	    //back side : symmetry
	    in  = grid->kend;
	    out = grid->kend+1;
	    u->new[i][j][out] = u->new[i][j][in]; //neumann
	    v->new[i][j][out] = v->new[i][j][in]; //neumann
        }
    }   	
    
    /* update ghost point at bottom & top */
    /* note you donot need to update v-velo, for it is zero at the face */
    for(i=grid->istart;i<= grid->iend;i++) {
        for(k=grid->kstart;k<= grid->kend;k++) { 
           //bottom face
	   in  = grid->jstart;
	   out = grid->jstart-1;
	   u->new[i][out][k] = -u->new[i][in][k]; //dirichlet,u=0
	   w->new[i][out][k] = -w->new[i][in][k]; //dirichlet,w=0
	   //top face
	   in  = grid->jend;
	   out = grid->jend+1;
	   u->new[i][out][k] = -u->new[i][in][k]; //dirichlet,u=0
	   w->new[i][out][k] = -w->new[i][in][k]; //dirichlet,w=0
        }
    }   	
   
    exchange_u(grid,u);
    
    if(Rank ==0) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
	        in  = grid->istart;
	        out = grid->istart-1;
	        v->new[out][j][k] = -v->new[in][j][k]; //dirichlet,v = 0
	        w->new[out][j][k] = -w->new[in][j][k]; //dirichlet,w = 0
            }
        }
    }
    /* right boundary */
    /* note v & w have already been calculated by outflow */
    
}


void update_phi_ghosts(Grid *grid,double ***phi)
{
    int i,j,k,in,out;
    void exchange_phi(Grid *,double ***);

     for(i=grid->istart;i<= grid->iend;i++) {
         for(j=grid->jstart;j<= grid->jend;j++) {
	     in  = grid->kstart;
	     out = grid->kstart-1;
	     phi[i][j][out]    = phi[i][j][in]; //neumann
	     in  = grid->kend;
	     out = grid->kend+1;
	     phi[i][j][out]    = phi[i][j][in]; //neumann
        }
    }   	

    for(i=grid->istart;i<= grid->iend;i++) {
        for(k=grid->kstart;k<= grid->kend;k++) { 
	    in  = grid->jstart;
	    out = grid->jstart-1;
	    phi[i][out][k]    = phi[i][in][k];  //neumann
	    in  = grid->jend;
	    out = grid->jend+1;
	    phi[i][out][k]    = phi[i][in][k];  //neumann
        }
    }   	

    exchange_phi(grid,phi);

    if(Rank ==0) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
	        in  = grid->istart;
	        out = grid->istart-1;
	        phi[out][j][k]    = phi[in][j][k]; //neumann
            }
        }
    }
  
    if(Rank == Size-1) {  
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
	        in  = grid->iend;
	        out = grid->iend + 1;
	        phi[out][j][k]    = phi[in][j][k]; //neumann
            }
        }
    }
 
}


void update_veloghost_points(Grid *grid, Velo *u,Velo *v,Velo *w)
{
    int i,j,k,in,out;
    void exchange_velo_boundary(Grid *,Velo*,Velo *,Velo *);
   
   
    /* update ghost point at side face*/
    /* note you donot need to update w-velo, for it is zero at the face*/
    for(i=grid->istart;i<= grid->iend;i++) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            //front side : symmetry
	    in  = grid->kstart;
	    out = grid->kstart-1;
	    u->new[i][j][out] = u->new[i][j][in]; //neumann
            v->new[i][j][out] = v->new[i][j][in]; //neumann
	    /* phi[i][j][out]    = phi[i][j][in]; */
	    //back side : symmetry
	    in  = grid->kend;
	    out = grid->kend+1;
	    u->new[i][j][out] = u->new[i][j][in];
	    v->new[i][j][out] = v->new[i][j][in];
	    /* phi[i][j][out]    = phi[i][j][in]; */
        }
    }   	
   
    /* update ghost point at bottom & top*/
    /* note you donot need to update v-velo, for it is zero at the face*/
    for(i=grid->istart;i<= grid->iend;i++) {
        for(k=grid->kstart;k<= grid->kend;k++) { 
            //bottom face
	    in  = grid->jstart;
	    out = grid->jstart-1;
	    u->new[i][out][k] = -u->new[i][in][k]; //dirichlet,u=0
            w->new[i][out][k] = -w->new[i][in][k]; //dirichlet,w=0
	    /* phi[i][out][k]    = phi[i][in][k];  */
	    //top face
	    in  = grid->jend;
	    out = grid->jend+1;
	    u->new[i][out][k] = -u->new[i][in][k]; //dirichlet,u=0
            w->new[i][out][k] = -w->new[i][in][k]; //dirichlet,w=0
	    /* phi[i][out][k]    = phi[i][in][k];  */
        }
    }   	
   
       
    /* first update the left & right boundary ghost points*/
    /* note: exchange of phi has been done */
    exchange_velo_boundary(grid,u,v,w);
   
    /* left boundary*/
    if(Rank ==0) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
	        in  = grid->istart;
	        out = grid->istart-1;
	        v->new[out][j][k] = -v->new[in][j][k]; //dirichlet,v = 0
	        w->new[out][j][k] = -w->new[in][j][k]; //dirichlet,w = 0
	        /* phi[out][j][k]    = phi[in][j][k]; */
            
            }
        }
    }
    /*
    if(Rank == Size-1) {  //right boundary
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
	        in  = grid->iend;
	        out = grid->iend + 1;
	        // note v & w have already been calculated by outflow 
	        phi[out][j][k]    = phi[in][j][k]; //neumann
            }
        }
    }
    */
 
}

void update_ghost_points(Grid *grid, double ***phi, Velo *u,Velo *v,Velo *w)
{
    int i,j,k,in,out;
    void exchange_velophi_boundary(Grid *,double***,Velo*,Velo *,Velo *);
   
   
    /* update ghost point at side face*/
    /* note you donot need to update w-velo, for it is zero at the face*/
    for(i=grid->istart;i<= grid->iend;i++) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            //front side : symmetry
	    in  = grid->kstart;
	    out = grid->kstart-1;
	    u->new[i][j][out] = u->new[i][j][in]; //neumann
            v->new[i][j][out] = v->new[i][j][in]; //neumann
	    phi[i][j][out]    = phi[i][j][in];
	    //back side : symmetry
	    in  = grid->kend;
	    out = grid->kend+1;
	    u->new[i][j][out] = u->new[i][j][in];
	    v->new[i][j][out] = v->new[i][j][in];
	    phi[i][j][out]    = phi[i][j][in];
        }
    }   	
   
    /* update ghost point at bottom & top*/
    /* note you donot need to update v-velo, for it is zero at the face*/
    for(i=grid->istart;i<= grid->iend;i++) {
        for(k=grid->kstart;k<= grid->kend;k++) { 
            //bottom face
	    in  = grid->jstart;
	    out = grid->jstart-1;
	    u->new[i][out][k] = -u->new[i][in][k]; //dirichlet,u=0
            w->new[i][out][k] = -w->new[i][in][k]; //dirichlet,w=0
	    phi[i][out][k]    = phi[i][in][k];
	    //top face
	    in  = grid->jend;
	    out = grid->jend+1;
	    u->new[i][out][k] = -u->new[i][in][k]; //dirichlet,u=0
            w->new[i][out][k] = -w->new[i][in][k]; //dirichlet,w=0
	    phi[i][out][k]    = phi[i][in][k];
        }
    }   	
   
       
    /* first update the left & right boundary ghost points*/
    /* note: exchange of phi has been done */
    exchange_velophi_boundary(grid,phi,u,v,w);
   
    /* left boundary*/
    if(Rank ==0) {
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
	        in  = grid->istart;
	        out = grid->istart-1;
	        v->new[out][j][k] = -v->new[in][j][k]; //dirichlet,v = 0
	        w->new[out][j][k] = -w->new[in][j][k]; //dirichlet,w = 0
	        phi[out][j][k]    = phi[in][j][k];
            
            }
        }
    }
    
    if(Rank == Size-1) {  //right boundary
        for(j=grid->jstart;j<= grid->jend;j++) {
            for(k=grid->kstart;k<= grid->kend;k++) {
	        in  = grid->iend;
	        out = grid->iend + 1;
	        /* note v & w have already been calculated by outflow */
	        phi[out][j][k]    = phi[in][j][k]; //neumann
            }
        }
    }
 
}


void pressure_gradient(double dt,double rk_gamma,double rk_rho,Grid *grid,double ***phi,Velo
*u,Velo *v,Velo *w)
{
  int i,j,k;
  double rk_alpha,fac,pyy,pxx,pzz;
  double ***phi_x,***phi_y,***phi_z;
  
  rk_alpha = rk_gamma + rk_rho;
  fac = 0.5*dt*rk_alpha;
  
  /* find the gradient of phi in y-dir at v location, including ghost point */ 
  phi_y = alloc3d_dble(v->nx_max,v->ny_max,v->nz_max);
  
  for(i=v->istart-1;i<= v->iend+1;i++) {
      for(j=v->jstart-1;j<= v->jend+1;j++) {
       for(k=v->kstart-1;k<= v->kend+1; k++) {
         phi_y[i][j][k] = (phi[i][j][k] - phi[i][j-1][k])/grid->dy;
      }
    } 
  }
  /* find new pressure gradient at v location(interior) */   
  for(i=v->istart;i<= v->iend;i++) {
      for(j=v->jstart;j<= v->jend;j++) {
       for(k=v->kstart;k<= v->kend; k++) {
        pxx = (v->Ax[i][j][k].a)*phi_y[i-1][j][k]+(v->Ax[i][j][k].b)*phi_y[i][j][k]+(v->Ax[i][j][k].c)*phi_y[i+1][j][k];
        pyy = (v->Ay[i][j][k].a)*phi_y[i][j-1][k]+(v->Ay[i][j][k].b)*phi_y[i][j][k]+(v->Ay[i][j][k].c)*phi_y[i][j+1][k];
        pzz = (v->Az[i][j][k].a)*phi_y[i][j][k-1]+(v->Az[i][j][k].b)*phi_y[i][j][k]+(v->Az[i][j][k].c)*phi_y[i][j][k+1];
        v->pg[i][j][k] = v->pg[i][j][k] + phi_y[i][j][k] -fac*(pyy+pxx+pzz);
      }
    }
  }   
  
  /* find the gradient of phi in x-dir at u-location*/
  phi_x = alloc3d_dble(u->nx_max,u->ny_max,u->nz_max); 
  
   for(i=u->istart-1;i<= u->iend+1;i++) {
      for(j=u->jstart-1;j<= u->jend+1;j++) {
       for(k=u->kstart-1;k<= u->kend+1; k++) {
        phi_x[i][j][k] = (phi[i][j][k] - phi[i-1][j][k])/grid->dx;
      }
    } 
  }
  /* find new pressure gradient at u location(interior) */ 
  for(i=u->istart;i<= u->iend;i++) {
      for(j=u->jstart;j<= u->jend;j++) {
       for(k=u->kstart;k<= u->kend; k++) {
        pxx = (u->Ax[i][j][k].a)*phi_x[i-1][j][k]+(u->Ax[i][j][k].b)*phi_x[i][j][k]+(u->Ax[i][j][k].c)*phi_x[i+1][j][k];
	pyy = (u->Ay[i][j][k].a)*phi_x[i][j-1][k]+(u->Ay[i][j][k].b)*phi_x[i][j][k]+(u->Ay[i][j][k].c)*phi_x[i][j+1][k];
        pzz = (u->Az[i][j][k].a)*phi_x[i][j][k-1]+(u->Az[i][j][k].b)*phi_x[i][j][k]+(u->Az[i][j][k].c)*phi_x[i][j][k+1];
        u->pg[i][j][k] = u->pg[i][j][k] + phi_x[i][j][k] -fac*(pyy+pxx+pzz);
     }
   } 
  }
  /* find the gradient of phi in z-dir at w-location*/
  phi_z = alloc3d_dble(w->nx_max,w->ny_max,w->nz_max); 
  
  for(i=w->istart-1;i<= w->iend+1;i++) {
      for(j=w->jstart-1;j<= w->jend+1;j++) {
       for(k=w->kstart-1;k<= w->kend+1; k++) {
        phi_z[i][j][k] = (phi[i][j][k] - phi[i][j][k-1])/grid->dz;
      }
    } 
  }
  /* find new pressure gradient at u location(interior) */ 
  for(i=w->istart;i<= w->iend;i++) {
      for(j=w->jstart;j<= w->jend;j++) {
       for(k=w->kstart;k<= w->kend; k++) {
        pxx = (w->Ax[i][j][k].a)*phi_z[i-1][j][k]+(w->Ax[i][j][k].b)*phi_z[i][j][k]+(w->Ax[i][j][k].c)*phi_z[i+1][j][k];
	pyy = (w->Ay[i][j][k].a)*phi_z[i][j-1][k]+(w->Ay[i][j][k].b)*phi_z[i][j][k]+(w->Ay[i][j][k].c)*phi_z[i][j+1][k];
        pzz = (w->Az[i][j][k].a)*phi_z[i][j][k-1]+(w->Az[i][j][k].b)*phi_z[i][j][k]+(w->Az[i][j][k].c)*phi_z[i][j][k+1];
        w->pg[i][j][k] = w->pg[i][j][k] + phi_z[i][j][k] -fac*(pyy+pxx+pzz);
      }
    }
  }  
    
 free3d_dble(u->nx_max,u->ny_max,u->nz_max,phi_x);
 free3d_dble(v->nx_max,v->ny_max,v->nz_max,phi_y);
 free3d_dble(w->nx_max,w->ny_max,w->nz_max,phi_z);
  
  
}


void write_to_file(int fileno,Grid *grid,double ***un, double ***vn,double ***wn,double ***cn)   
{
    int i,j,k,m;
    float data[4];
    double uval,vval,wval,cval;
    char num[6],b[6],fn[20]="caf";
    char dat[] = ".bin";
    FILE *fp;

    sprintf(b,"%04d",fileno);
    strcat(fn,b);
    
    sprintf(num,"R%03d",Rank);
    strcat(fn,num);
    
    strcat(fn,dat);
  
    
    fp = fopen(fn,"wb");
    fclose(fp);

    fp = fopen(fn,"ab");
    for(i=grid->istart;i<= grid->iend;i++) {
    	for(j=grid->jstart;j<= grid->jend;j++) {
    	    for(k=grid->kstart;k<= grid->kend;k++) {
    		uval = 0.5*(un[i][j][k]+un[i+1][j][k]);
    		vval = 0.5*(vn[i][j][k]+vn[i][j+1][k]);
    		wval = 0.5*(wn[i][j][k]+wn[i][j][k+1]);
    		cval = cn[i][j][k];

    		data[0] = (float)uval;
    		data[1] = (float)vval;
    		data[2] = (float)wval;
    		data[3] = (float)cval;
		if(Nx_cell_global*Ny_cell_global*Nz_cell_global > 2e7){
		    if(i%2==0 & j%2==0 & k%2==0)fwrite(data,sizeof(float),4,fp);
		}else{
		    fwrite(data,sizeof(float),4,fp);
		}
    	    }
    	}
    }
    fclose(fp);
  	
  
}


double find_cfl(Grid *grid,double ***un,double ***vn,double ***wn)
{
  int i,j,k;
  double uval,vval,wval,cfl,maxval,cfldt,maxall;

  maxval = 0.0;
  for(i=grid->istart;i<= grid->iend;i++) {
     for(j=grid->jstart;j<= grid->jend;j++) {
        for(k=grid->kstart;k<= grid->kend;k++) {
	   uval = 0.5*(un[i][j][k]+ un[i+1][j][k]);
	   vval = 0.5*(vn[i][j][k]+ vn[i][j+1][k]);
	   wval = 0.5*(wn[i][j][k]+ wn[i][j][k+1]);
	   cfl = (fabs(uval)/grid->dx) + (fabs(vval)/grid->dy)+(fabs(wval)/grid->dz);
	   maxval = (cfl > maxval)? cfl : maxval;
	} 
     }	  
  }
  
  MPI_Allreduce(&maxval,&maxall,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); 

  cfldt = CFLmax/maxall;
    	
  return cfldt;
}  


void midfield(int tstep,int fileno,double tval,double outtime,double outmid, double cavgtime, Grid *grid,Conc *c,Velo *u,Velo*v,Velo*w)
{
   int i,j,m,nx,ny,nz;
   char num[6],fn[20]="midfield";
   char dat[] = ".bin";
   
   FILE *fp1,*fp2;
  
   if(Rank == 0) {
      fp1 = fopen("middata","w");
      fprintf(fp1,"%d\n",tstep);
      fprintf(fp1,"%d\n",fileno);
      fprintf(fp1,"%lf\n",tval);
      fprintf(fp1,"%lf\n",outtime);
      fprintf(fp1,"%lf\n",outmid);
      fprintf(fp1,"%lf\n",cavgtime);
    
      fclose(fp1);
  
      
   }
   
   sprintf(num,"%03d",Rank);
   strcat(fn,num);
   strcat(fn,dat);
   
   fp2 = fopen(fn,"wb");
   fclose(fp2);
  
   nx = grid->nx_max; // array is assumed equal for all structures
   ny = grid->ny_max;
   nz = grid->nz_max;
 

   fp2 = fopen(fn,"ab");
   for(i=0;i< nx;i++) {
      for(j=0;j< ny;j++) {
   	 fwrite(c->old[i][j],sizeof(double),nz,fp2);
   	 fwrite(c->explicit_nm1[i][j],sizeof(double),nz,fp2);
   	 fwrite(u->old[i][j],sizeof(double),nz,fp2);
   	 fwrite(u->new[i][j],sizeof(double),nz,fp2);
   	 fwrite(u->explicit_nm1[i][j],sizeof(double),nz,fp2);
   	 fwrite(u->pg[i][j],sizeof(double),nz,fp2);
   	 fwrite(v->old[i][j],sizeof(double),nz,fp2);
   	 fwrite(v->new[i][j],sizeof(double),nz,fp2);
   	 fwrite(v->explicit_nm1[i][j],sizeof(double),nz,fp2);
   	 fwrite(v->pg[i][j],sizeof(double),nz,fp2);
   	 fwrite(w->old[i][j],sizeof(double),nz,fp2);
   	 fwrite(w->new[i][j],sizeof(double),nz,fp2);
   	 fwrite(w->explicit_nm1[i][j],sizeof(double),nz,fp2);
   	 fwrite(w->pg[i][j],sizeof(double),nz,fp2);

      }    
   }
   fclose(fp2);
	
     	
}


void intermediate(int *tstep,int *fileno,double *tval,double *outtime,Grid *grid,Conc *c,Velo *u,Velo*v,Velo*w)
{
    int i,j,m,nx,ny,nz;
    //int ptr_location;
    char num[6],fn[20]="midfield";
    char dat[] = ".bin";
    
    FILE *fp1,*fp2;
  
    fp1 = fopen("middata","r");
        fscanf(fp1,"%d\n",tstep);
        fscanf(fp1,"%d\n",fileno);
        fscanf(fp1,"%lf\n",tval);
        fscanf(fp1,"%lf\n",outtime);
    fclose(fp1);
  
    nx = grid->nx_max; // array is assumed equal for all structures
    ny = grid->ny_max;
    nz = grid->nz_max;
    
    sprintf(num,"%03d",Rank);
    strcat(fn,num);
    strcat(fn,dat);
  
    fp2 = fopen(fn,"rb");
    //ptr_location = nx*ny*14*nz*m;
    //fseek(fp2,ptr_location*sizeof(double),SEEK_SET);
    for(i=0;i< nx;i++) {
        for(j=0;j< ny;j++) {
   	    fread(c->old[i][j],sizeof(double),nz,fp2);
   	    fread(c->explicit_nm1[i][j],sizeof(double),nz,fp2);
   	    fread(u->old[i][j],sizeof(double),nz,fp2);
   	    fread(u->new[i][j],sizeof(double),nz,fp2);
   	    fread(u->explicit_nm1[i][j],sizeof(double),nz,fp2);
   	    fread(u->pg[i][j],sizeof(double),nz,fp2);
   	    fread(v->old[i][j],sizeof(double),nz,fp2);
   	    fread(v->new[i][j],sizeof(double),nz,fp2);
   	    fread(v->explicit_nm1[i][j],sizeof(double),nz,fp2);
   	    fread(v->pg[i][j],sizeof(double),nz,fp2);
   	    fread(w->old[i][j],sizeof(double),nz,fp2);
   	    fread(w->new[i][j],sizeof(double),nz,fp2);
   	    fread(w->explicit_nm1[i][j],sizeof(double),nz,fp2);
   	    fread(w->pg[i][j],sizeof(double),nz,fp2);
        }    
    }
    fclose(fp2);
    	
}


void shift_conc(Grid *grid,Conc *c)
{
   int i, j, k, fno;
   double beta, meshdx;
   double *x, *xx,  *y, *yy, *f ; 
   
    x  = (double*)calloc(grid->nx_max,sizeof(double));  
    xx = (double*)calloc(grid->nx_max,sizeof(double));
    yy = (double*)calloc(grid->nx_max,sizeof(double));   // c->old in new mesh, calculated by interpolation
    y  = (double*)calloc(grid->nx_max,sizeof(double));
    f  = (double*)calloc(grid->nx_max,sizeof(double));
  
    for(k=0;k<grid->nz_max;k++) {
    
        beta = 2*PI/Lamb;
	
	if(Subharm == 0){
            meshdx = Ampli*cos(beta*grid->Zc[k]);      // mesh translation
	}
	else if(Subharm == 1){
	    meshdx = Ampli*cos(beta*(grid->Zc[k]+Lamb/2)) + Ampli*cos(beta/2*(grid->Zc[k]+Lamb/2));
	}
	
                          
        for(j=0;j<grid->ny_max;j++) {
	
            for(i=0;i<grid->nx_max;i++) {
	        y[i] = c->old[i][j][k];
	    	x[i] = grid->Xc[i];
	    	xx[i] = x[i] - meshdx;
	    }
	   
	    spline(x,xx,y,yy,grid);
	    			    
	    for(i=0;i<grid->nx_max;i++) { 
	    	c->old[i][j][k] = yy[i]; 
	    }
	    
	}
    }

    free(x);
    free(xx);
    free(y);
    free(yy);
    free(f);

}

void spline(double *x, double *xx, double *y, double *yy, Grid *grid)
{

    int i, ii, N;
    double delta;
    double *fpp;
    Tds *tri_fpp;
    
    N = grid->nx_max;
    delta = grid->dx;
 
    fpp = (double*)calloc(N,sizeof(double)); 
    tri_fpp = alloc_tds(1,N); 
    
    /* TDS for the second derivative */

    for(i=1;i<N-1;i++) {
        tri_fpp->acof[i-1] = 1.0;
        tri_fpp->bcof[i-1] = 4.0;
        tri_fpp->ccof[i-1] = 1.0;
        tri_fpp->rhs[i-1] = 6.0/delta/delta*( y[i+1] -2.0*y[i] + y[i-1] );
	
    }

    if(Rank == 0){
        fpp[0] = 0;
    }
    if(Rank == Size-1){
        fpp[N-1] = 0;
    }

    pll_tridiag(tri_fpp,1,N-2);
    //tridiag_1rhs(N-2,tri_fpp->acof,tri_fpp->bcof,tri_fpp->ccof,tri_fpp->rhs);

    for(i=1;i<N-1;i++) {
        fpp[i] = tri_fpp->rhs[i-1];
    }   

    /* Find the proper interval and calculate the polynomial */
    for(ii=0;ii<N;ii++) { 

        for(i=0;i<N-1;i++) {

	    if(x[i] <= xx[ii] && xx[ii] <= x[i+1]){
	   
	        yy[ii] = fpp[ i ]*( x[i+1]-xx[ii] )*( x[i+1]-xx[ii] )*( x[i+1]-xx[ii] )/(6*delta)
	               + fpp[i+1]*( xx[ii] - x[i] )*( xx[ii] - x[i] )*( xx[ii] - x[i] )/(6*delta)
	               + ( y[ i ]/delta - delta/6*fpp[ i ] )*( x[i+1]-xx[ii] )
	               + ( y[i+1]/delta - delta/6*fpp[i+1] )*( xx[ii] - x[i] );
	       	      
            }
        }
       
        if(xx[ii] > x[N-1]){
            yy[ii] = y[N-1];
        }	   
        else if(xx[ii] < x[1]){
            yy[ii] = y[0];
        }   
    }

    free(fpp);
    free_tds(tri_fpp,1,N);

}


void print_to_file(double *y, Grid *grid, int fno)
{

    int i,m;
    float data;
    char b[6],fn[20]="cval";
    char dat[] = ".bin";
    FILE *fp;

    sprintf(b,"%02d",fno);
    strcat(fn,b);
    strcat(fn,dat);
  
    if(Rank == 0 ) {
       fp = fopen(fn,"wb");
       fclose(fp);
    }  

    for(m=0;m<Size;m++) {
  
    	if(Rank == m) {
            fp = fopen(fn,"ab");
            for(i=grid->istart;i<= grid->iend;i++) {	 
        	data = (float)y[i];
        	fwrite(&data,sizeof(float),1,fp);
            }
            fclose(fp);
        }
        MPI_Barrier(MPI_COMM_WORLD);
	
    }

}

double bulk_velo(Grid *grid,double ***un)
{
   int j,k;
   double vbulk;
   
   vbulk = 0.0;
   
   for(j=grid->jstart;j<= grid->jend;j++) {
      for(k=grid->kstart;k<= grid->kend;k++) {
         if(fabs(un[grid->iend][j][k]) > fabs(vbulk)) { 
             vbulk = un[grid->iend][j][k];
	 }    
      }	    
   }
   

  return vbulk;
}



void update_old_conc(Conc *c)
{
   int i,j,nz;
   
   nz = c->nz_max;
   
   for(i=0;i< c->nx_max;i++) {
      for(j=0;j< c->ny_max;j++) {
         memcpy(c->old[i][j],         c->new[i][j],       nz*sizeof(double));
	 memcpy(c->explicit_nm1[i][j],c->explicit_n[i][j],nz*sizeof(double));
     }    	 
  }
}


void update_old_velo(Velo *u,Velo *v,Velo *w)
{
   int i,j,nz;
   
   nz = u->nz_max;  // note: it is assumed that all velo structures are of equal Size
   
   for(i=0;i< u->nx_max;i++) {
      for(j=0;j< u->ny_max;j++) {
         memcpy(u->old[i][j],         u->new[i][j],       nz*sizeof(double));   // data value
	 memcpy(u->explicit_nm1[i][j],u->explicit_n[i][j],nz*sizeof(double));   // conv&explicit
	 memcpy(v->old[i][j],         v->new[i][j],       nz*sizeof(double)); 
	 memcpy(v->explicit_nm1[i][j],v->explicit_n[i][j],nz*sizeof(double)); 
	 memcpy(w->old[i][j],         w->new[i][j],       nz*sizeof(double));
	 memcpy(w->explicit_nm1[i][j],w->explicit_n[i][j],nz*sizeof(double));
     }    	 
   }
}


void update_old_visc(Visc *mu_old,Visc *mu_new)
{
   int i,j,nx,ny,nz;
   
   nx = mu_new->nx_max;
   ny = mu_new->ny_max;
   nz = mu_new->nz_max;
   
   for(i=0;i< nx;i++) {
      for(j=0;j< ny;j++) {
         memcpy(mu_old->cell[i][j],mu_new->cell[i][j],nz*sizeof(double));
	 memcpy(mu_old->xy[i][j]  ,mu_new->xy[i][j]  ,nz*sizeof(double));
	 memcpy(mu_old->xz[i][j]  ,mu_new->xz[i][j]  ,nz*sizeof(double));
	 memcpy(mu_old->yz[i][j]  ,mu_new->yz[i][j]  ,nz*sizeof(double));
     }    	 
  }
}


double calculate_weno(double *d1,int kk,int sign)
{
    int i,j0,j1,j2,j3,j4;
    double con1,con2,con3,d2,maxd;
    double c1,c2,c3,t1,t2,s1,s2,s3;
    double epi,alp1,alp2,alp3,alpt,w1,w2,w3,deriv;

    con1 = 1.0/6.0;
    con2 = 13.0/12.0;
    con3 = 1.0/4.0; 

    j0 = 0;
    j1 = sign;
    j2 = sign*2;
    j3 = sign*3;
    j4 = sign*4;

    maxd = 0.0;
    for(i=0;i<5;i++) {
        d2 = d1[kk+sign*i] * d1[kk+sign*i];
        maxd = (maxd > d2) ? maxd : d2;
    }
    
    c1 = con1*( 2.0*d1[kk+j0] - 7.0*d1[kk+j1] + 11.0*d1[kk+j2]);
    c2 = con1*(-1.0*d1[kk+j1] + 5.0*d1[kk+j2] +  2.0*d1[kk+j3]);
    c3 = con1*( 2.0*d1[kk+j2] + 5.0*d1[kk+j3] -  1.0*d1[kk+j4]);

    t1 = d1[kk+j0] - 2.0*d1[kk+j1] + 1.0*d1[kk+j2];
    t2 = d1[kk+j0] - 4.0*d1[kk+j1] + 3.0*d1[kk+j2];
    s1 = (con2*t1*t1) + (con3*t2*t2);

    t1 = d1[kk+j1] - 2.0*d1[kk+j2] + d1[kk+j3];
    t2 = d1[kk+j1] - d1[kk+j3];
    s2 = (con2*t1*t1) + (con3*t2*t2);

    t1 = 1.0*d1[kk+j2] - 2.0*d1[kk+j3] + d1[kk+j4];
    t2 = 3.0*d1[kk+j2] - 4.0*d1[kk+j3] + d1[kk+j4];
    s3 = (con2*t1*t1) + (con3*t2*t2);

    epi = 1.0e-6*maxd + 1.0e-99;
    alp1 = 0.1/((s1+epi)*(s1+epi));
    alp2 = 0.6/((s2+epi)*(s2+epi));
    alp3 = 0.3/((s3+epi)*(s3+epi));
    alpt = alp1 + alp2 + alp3;

    w1  = alp1/alpt;
    w2  = alp2/alpt;
    w3  = alp3/alpt;

    deriv = w1*c1 + w2*c2 + w3*c3;

    return deriv;
}


/* function originally defined in banded_solvers.c */
int tridiag_1rhs(int Nmax,double *a,double *b,double *c,double *rhs)
{
   int i;
   double bb,*cc;

   cc = alloc1d_dble(Nmax);

   bb = b[0]; 
   cc[0] = c[0]/bb ;
   rhs[0] = rhs[0]/bb;

   for(i=1;i<Nmax;i++) {
      bb = b[i] - cc[i-1]*a[i];
      cc[i] = c[i]/bb;

      rhs[i] = (rhs[i] - rhs[i-1]*a[i])/bb; 	 
   } 

   for(i=Nmax-2;i>=0;i--) {
      rhs[i] = rhs[i] - rhs[i+1]*cc[i];
   }   

   free(cc);
   return 0;
 
}


void find_max(Grid* grid,Velo *u,Velo *v,Velo *w,Conc *c)
{
    int i,j,k,ui,uj,uk,vi,vj,vk,wi,wj,wk,ci,cj,ck;
    double a1,b1,c1;
    double maxu=0,maxv=0,maxw=0,maxc=0;
    
    FILE *fp;
	
    for(i=0;i< grid->nx_max;i++) {
	for(j=0;j< grid->ny_max;j++) {
	    for(k=0;k< grid->nz_max;k++) {

	       if(fabs(maxu) <= fabs(u->new[i][j][k])) {
	           maxu = u->new[i][j][k];
		   ui = i; uj = j; uk = k;
	       }
	       if(fabs(maxv) <= fabs(v->new[i][j][k])) {
	           maxv = v->new[i][j][k];
		   vi = i; vj = j; vk = k;
	       }
	       if(fabs(maxw) <= fabs(w->new[i][j][k])) {
	           maxw = w->new[i][j][k];
		   wi = i; wj = j; wk = k;
	       }
               if(fabs(maxc) <= fabs(c->new[i][j][k])) {
	           maxc = c->new[i][j][k];
		   ci = i; cj = j; ck = k;
	       }

               a1 = c->old[i][j][k];
	       if(isnan(a1)) {
	           fp = fopen("Outscreen","a");
                   fprintf(fp,"nan in a %d %d %d \n",i,j,k);
		   fclose(fp);
		   exit(1);
	       }
	       b1 = c->new[i][j][k];
	       if(isnan(b1)) {
	           fp = fopen("Outscreen","a");
                   fprintf(fp,"nan in b %d %d %d \n",i,j,k);
		   fclose(fp);
		   exit(1);
	       }
 	       c1 = c->explicit_n[i][j][k];
	       if(isnan(c1)) {
	           fp = fopen("Outscreen","a");
                   fprintf(fp,"nan in c %d %d %d \n",i,j,k);
		   fclose(fp);
		   exit(1);
	       }

            }
	}
    }
    
    
    if(Rank == 0){
       fp = fopen("Outscreen","a");
       fprintf(fp,"max\n");
       fprintf(fp,"  %2d %2d %2d %lf\n  %2d %2d %2d %lf\n  %2d %2d %2d %lf\n  %2d %2d %2d %lf\n\n",
    	   				ui,uj,uk,maxu,vi,vj,vk,maxv,wi,wj,wk,maxw,ci,cj,ck,maxc);
       fclose(fp);				
    }					
    
}


void find_wmax(double tval, Grid *grid, Velo *w)
{
   int i,j,k;
   double wmax,wmaxall;
   float wdata[2];
   FILE *fp;
   
   wmax = 0.0;
   for(i=grid->istart;i<= grid->iend;i++) {
       for(j=grid->jstart;j<= grid->jend;j++) {
           for(k=grid->kstart;k<= grid->kend;k++) {
	       wmax = (w->new[i][j][k] > wmax)? w->new[i][j][k] : wmax;
	   }
       }	  
   }
  
   MPI_Reduce(&wmax,&wmaxall,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); 

   if(Rank == 0){
       wdata[0] = (float)wmaxall;
       wdata[1] = (float)tval;
       fp = fopen("wmax.bin","ab");
       fwrite(&wdata,sizeof(float),2,fp);
       fclose(fp);
   }    

}


void conc_average(Grid *grid, double ***cn, double tval)
{
    int i, j, k, m, index;
    float data;
    double sumz, sumy, sumyz, avgy, avgz, avgyz;
    double *cyz;
    FILE *fp;
    
    cyz = alloc1d_dble(grid->nx_cell);
    
    /* average in y */    
    for(m=0;m<Size;m++) {
        if(Rank == m) {
            fp = fopen("cavgy.bin","ab");
            for(i=grid->istart;i<= grid->iend;i++) {
	        for(k=grid->kstart;k<=grid->kend;k++){
		
		    sumy = 0;
		    for(j=grid->jstart;j<=grid->jend;j++){
	                sumy += cn[i][j][k];
	            }
	            avgy = sumy/grid->ny_cell;
                    data = (float)avgy;
                    fwrite(&data,sizeof(float),1,fp);
		    
		}
            }
            fclose(fp);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    
    /* average in z */
    for(m=0;m<Size;m++) {
        if(Rank == m) {
            fp = fopen("cavgz.bin","ab");
            for(i=grid->istart;i<= grid->iend;i++) {
	        for(j=grid->jstart;j<=grid->jend;j++){
		    sumz = 0;
		    for(k=grid->kstart;k<=grid->kend;k++){
	                sumz += cn[i][j][k];
	            }
	            avgz = sumz/grid->nz_cell;
                    data = (float)avgz;
                    fwrite(&data,sizeof(float),1,fp);
		    
		    
		}
            }
            fclose(fp);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    
    /* average in both y and z */
    index = 0;
    sumyz = 0;
    for(i=grid->istart;i<=grid->iend;i++){
        for(j=grid->jstart;j<=grid->jend;j++){
	    for(k=grid->kstart;k<=grid->kend;k++){
	        sumyz += cn[i][j][k];
	    }
	}
	avgyz = sumyz/(grid->ny_cell*grid->nz_cell);
	cyz[index] = avgyz;
	index++;
	sumyz = 0;
    }
    
    if(Rank == 0) {
        fp = fopen("cavgyz.bin","ab");
	data = (float)tval;
	fwrite(&data,sizeof(float),1,fp);
	fclose(fp);
    }
    for(m=0;m<Size;m++) {
        if(Rank == m) {
            fp = fopen("cavgyz.bin","ab");
            for(i=grid->istart;i<= grid->iend;i++) {
	    
                data = (float)cyz[i-grid->istart];
                fwrite(&data,sizeof(float),1,fp);
		
            }
            fclose(fp);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    free(cyz);
    
}





