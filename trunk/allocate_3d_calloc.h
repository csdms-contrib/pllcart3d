/*Header file containing all the allocating functions */
#ifndef ALLOCATE_H
#define ALLOCATE_H


/*Allocate 1-d array of type double */
double* alloc1d_dble(int n)
{
  double *array1 = (double*)calloc(n,sizeof(double));
 	 
  return array1;
} 

/* allocate 2-d array of type int */    
int** alloc2d_int(int nrows,int ncols)
{
   int i;
   
   int **array2 = (int **)malloc(nrows*sizeof(int *));

   array2[0] = (int *)calloc(nrows * ncols,sizeof(int));
   for(i = 1; i < nrows; i++) {
       array2[i] = array2[0] + i * ncols;
   } 
   
   return array2;
}  

/* allocate 2-d array of type double */    
double** alloc2d_dble(int nrows,int ncols)
{
   int i;
   
   double **array2 = (double **)malloc(nrows*sizeof(double *));

   array2[0] = (double *)calloc(nrows * ncols,sizeof(double));
   for(i = 1; i < nrows; i++) {
       array2[i] = array2[0] + i * ncols;
   } 
   
   return array2;
}      


/* allocate 3-d array of type double */    
double*** alloc3d_dble(int nrows,int ncols,int nz)
{
   int i,j;
   
   /* 3-d pointer */
   double ***array3 = (double ***)malloc(nrows *sizeof(double **));

   for(i = 0; i < nrows; i++) {
       /* 2-d pointer */
       array3[i] = (double **)malloc(ncols*sizeof(double*));
       for(j=0;j<ncols;j++) {
          /* 1-d pointer */
          array3[i][j] = (double *)calloc(nz,sizeof(double));
	  
       }
      
   } 
   return array3;
}  


/* allocate the tridiagonal structure with nrows and each struct with ncols coeffs */
Tds* alloc_tds(int nrows,int ncols)
{
  int i;
  Tds *arr = (Tds*)malloc(sizeof(Tds)*nrows);
  
  for(i=0;i<nrows;i++) {
     arr[i].acof = (double*)calloc(ncols,sizeof(double));
     arr[i].bcof = (double*)calloc(ncols,sizeof(double));
     arr[i].ccof = (double*)calloc(ncols,sizeof(double));
     arr[i].rhs  = (double*)calloc(ncols,sizeof(double));
  }
  
  return arr;
  
} 

/* allocate 3-d array of type Tri_abc */    
Tri_abc*** alloc3d_tri_abc(int nrows,int ncols,int nz)
{
   int i,j;
      
   /* 3-d pointer */
   Tri_abc ***array3 = (Tri_abc ***)malloc(nrows * sizeof(Tri_abc **));

   for(i = 0; i < nrows; i++) {
       /* 2-d pointer */
       array3[i] = (Tri_abc **)malloc(ncols * sizeof(Tri_abc*));
       for(j=0;j<ncols;j++) {
          /* 1-d pointer */
          array3[i][j] = (Tri_abc *)malloc(nz * sizeof(Tri_abc));
	  
       }
      
   } 
   
   return array3;
} 

/* allocate momentum(velocity) variables */
Velo* alloc_velo(int nrows,int ncols,int nz)
{
  Velo* var;
  
  var = (Velo*) malloc(1*sizeof(Velo));
  var->nx_max = nrows;
  var->ny_max = ncols;
  var->nz_max = nz;
  var->old      = alloc3d_dble(nrows,ncols,nz);
  var->new      = alloc3d_dble(nrows,ncols,nz);
  var->pg       = alloc3d_dble(nrows,ncols,nz);
  var->explicit_nm1 = alloc3d_dble(nrows,ncols,nz);
  var->explicit_n = alloc3d_dble(nrows,ncols,nz);
  var->Ax       = alloc3d_tri_abc(nrows,ncols,nz);
  var->Ay       = alloc3d_tri_abc(nrows,ncols,nz);
  var->Az       = alloc3d_tri_abc(nrows,ncols,nz);
    
  return var;
}

/* allocate concentration variables */
Conc* alloc_conc(int nrows,int ncols,int nz)
{
  Conc* var;
  
  var = (Conc*) malloc(1*sizeof(Conc));
  var->nx_max = nrows;
  var->ny_max = ncols;
  var->nz_max = nz;
  var->old      = alloc3d_dble(nrows,ncols,nz);
  var->new      = alloc3d_dble(nrows,ncols,nz);
  var->explicit_nm1 = alloc3d_dble(nrows,ncols,nz);
  var->explicit_n = alloc3d_dble(nrows,ncols,nz);
  var->Ax       = alloc3d_tri_abc(nrows,ncols,nz);
  var->Ay       = alloc3d_tri_abc(nrows,ncols,nz);
  var->Az       = alloc3d_tri_abc(nrows,ncols,nz);
  
  return var;
}

/* allocate concentration variables */
Visc* alloc_visc(int nrows,int ncols,int nz)
{
  Visc* var;
  
  var = (Visc*) malloc(1*sizeof(Visc));
  
  var->nx_max = nrows;
  var->ny_max = ncols;
  var->nz_max = nz;
  
  /*value at cell center */
  var->cell = alloc3d_dble(nrows,ncols,nz);
  /*value at nodal points in each plane */
  var->xy   = alloc3d_dble(nrows,ncols,nz);
  var->xz   = alloc3d_dble(nrows,ncols,nz);
  var->yz   = alloc3d_dble(nrows,ncols,nz);
  
  return var;
}


/* free the above allocated 3-d array of type double */
void free3d_dble(int nrows,int ncols,int nz,double ***array3)
{
  int i,j;
  
  for(i=0;i<nrows;i++) {
     for(j=0;j<ncols;j++) {
       free(array3[i][j]);
     } 
     
     free(array3[i]);
  }    	 
  free(array3);
}


/* free the above allocated 2-d array of type double */
void free2d_dble(double **array2)
{
  free(array2[0]);
  free(array2);
}

void free_tds(Tds *arr,int nrows,int ncols)
{
  int i;
  /* free the elements  of the structure first */
  for(i=0;i<nrows;i++) {
     free(arr[i].acof);
     free(arr[i].bcof);
     free(arr[i].ccof);
     free(arr[i].rhs);
  }
  
  free(arr);
}
#endif  
