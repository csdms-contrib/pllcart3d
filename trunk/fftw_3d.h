/* FFT routines - refer fftw manual */
/* Declarations for real cosine fft */
void plan_dct_3d(int,int,int);
void fdct_3d(int,int,int,double***,double***);
void idct_3d(int,int,int,double***,double***);

/*Global dct variables */
static fftw_plan DCT_fplan_3d,DCT_iplan_3d;
static double *DCT_in_3d,*DCT_out_3d;

/*planner for dct */
void  plan_dct_3d(int ndim1,int ndim2,int num_transforms)
{
   int N,Nt;
   int N_arr[2];
   int istride,ostride,idist,odist;
   int *inembed=NULL,*onembed=NULL;
   fftw_r2r_kind fkind[2]={FFTW_REDFT10,FFTW_REDFT10};
   fftw_r2r_kind ikind[2]={FFTW_REDFT01,FFTW_REDFT01};
   
   N  = ndim1*ndim2;
   Nt = num_transforms;
   N_arr[0] = ndim2;
   N_arr[1] = ndim1;
   
   /*advanced interface- manum_transforms dfts*/
   istride =1;
   ostride =1;
   idist = N;
   odist = N;
   DCT_in_3d  = (double*)fftw_malloc(N*Nt*sizeof(double));
   DCT_out_3d = (double*)fftw_malloc(N*Nt*sizeof(double));

  /*manum_transforms ffts - forward dct transform */ 
  DCT_fplan_3d =fftw_plan_many_r2r(2,N_arr,Nt,DCT_in_3d,inembed,istride,idist,DCT_out_3d,onembed,ostride,odist,fkind,FFTW_MEASURE);
  /*manum_transforms iffts - inverse dct transform */ 
  DCT_iplan_3d =fftw_plan_many_r2r(2,N_arr,Nt,DCT_out_3d,inembed,istride,idist,DCT_in_3d,onembed,ostride,odist,ikind,FFTW_MEASURE);
                                    
}

/*function to find the 1-dimensional dct of a 2-d array */
/* Note: Array for each FFT transform should be contiguous. (i.e.) input format should be of
datain[array_size][number_of_transforms]. FFTW then transforms each column*/
 
void fdct_3d(int ndim1,int ndim2,int num_transforms,double ***datain,double ***dataout)
{

  int i,j,k,N1,N2,N,Nt;
  
  N1 = ndim1;
  N2 = ndim2;
  Nt = num_transforms;
  N = N1*N2;
   
  for(k=0;k<Nt;k++) {
    for(j=0;j<N2;j++) {
      for(i=0;i<N1;i++) {
         DCT_in_3d[k*N+j*N1+i] = datain[i][j][k];
      }	 
    }
  } 
  
  fftw_execute(DCT_fplan_3d);
  
  for(k=0;k<Nt;k++) {
    for(j=0;j<N2;j++) {
      for(i=0;i<N1;i++) {
        dataout[i][j][k] = DCT_out_3d[k*N+j*N1+i]; 
      }	
    }
  }
 
 /*end of function */            
} 

/*function to find the inverse 1-dimensional idct of a 2-d array */
void idct_3d(int ndim1,int ndim2,int num_transforms,double ***datain,double ***dataout)
{

  int i,j,k,N1,N2,N,Nt;
  
  N1 = ndim1;
  N2 = ndim2;
  Nt = num_transforms;
  N = N1*N2;
   
  for(k=0;k<Nt;k++) {
    for(j=0;j<N2;j++) {
      for(i=0;i<N1;i++) {
         DCT_out_3d[k*N+j*N1+i] = datain[i][j][k];
      }
    }
  }    	 
    
  fftw_execute(DCT_iplan_3d);
  
  for(k=0;k<Nt;k++) {
    for(j=0;j<N2;j++) {
      for(i=0;i<N1;i++) {
        /*Normalize */
        dataout[i][j][k] = DCT_in_3d[k*N+j*N1+i]/(double)(4*N); 
     }  
    }
  }
             
} 


