#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <math.h>

#define FORWARD 1
#define BACKWARD 2

extern int Rank;
extern int Size;

typedef struct tridiagonal_matrix_system
{
 double *acof;
 double *bcof;
 double *ccof;
 double *rhs;
}Tds;


void forward_compute(double *a, double *b, double *c, double *r,int n,double *recv,double *send)
{
 int i;
 double c_minus1,r_minus1;
 
 
 c_minus1 = recv[0];
 r_minus1 = recv[1];
 
 /* first row */
 b[0] = b[0] - c_minus1*a[0];
 c[0] = c[0]/b[0];
 r[0] = (r[0] - r_minus1*a[0])/b[0];
 
 /* rest of the row */
 for(i=1;i<n;i++)
 {
  
  b[i]=(b[i]-c[i-1]*a[i]);
  c[i]=c[i]/b[i];
  r[i]=(r[i]-r[i-1]*a[i])/b[i];
 }

 send[0]=c[n-1];
 send[1]=r[n-1];
 
}

void forward(Tds *arr,int N,int n)
{
   int i,ierr;
   int left_proc,right_proc,send_flag = 0;
   double send_buffer[2],recv_buffer[2],tmp_send_buffer[2];
   MPI_Status status; 
   MPI_Request request;
   
   left_proc  = (Rank == 0)? MPI_PROC_NULL : Rank - 1;
   right_proc = (Rank == Size-1)? MPI_PROC_NULL : Rank + 1;
   
   /* do forward elimination of all matrices */
   for(i=0;i<N;i++)
   {
      /* initiate the receive command for all processes except first */
      if(Rank!=0)
      {
         MPI_Recv(recv_buffer,2,MPI_DOUBLE,left_proc,FORWARD,MPI_COMM_WORLD,&status);
       
      } 
      else
      {
         recv_buffer[0] = 0.0;
	 recv_buffer[1] = 0.0;
      }
     
      /* do forward elimination & the end-points to be send */  
      forward_compute(arr[i].acof,arr[i].bcof,arr[i].ccof,arr[i].rhs,n,recv_buffer,tmp_send_buffer);
      
      /* wait & make sure that previous send command is complete */
      if(send_flag) {
         ierr = MPI_Wait(&request,&status);
      }
      send_buffer[0] = tmp_send_buffer[0];
      send_buffer[1] = tmp_send_buffer[1];
      
      /* Initiate a non-blocking send */
      ierr = MPI_Isend(send_buffer,2, MPI_DOUBLE,right_proc,FORWARD,MPI_COMM_WORLD,&request); 
      
      /*set the send flag to 1*/
      send_flag = 1;
      
   }

  
}

double  backward_compute(double *c, double *r,int n,double recv)
{
  int i;
  
   /*last row solution */
   r[n-1] = r[n-1] - c[n-1]*recv;
   
   for(i=n-2;i>=0;i--)
   {
      r[i]=r[i]-c[i]*r[i+1];
   }
   
   return r[0];
  
}


void backward(Tds *arr,int N, int n)
{
 
  int i,ierr,send_flag = 0;
  int left_proc,right_proc;
  double recv_buffer,tmp_send_buffer,send_buffer;
  MPI_Status status; 
  MPI_Request request;
  
  left_proc  = (Rank == 0)? MPI_PROC_NULL : Rank - 1;
  right_proc = (Rank == Size-1)? MPI_PROC_NULL : Rank + 1;
  
  for(i=0;i<N;i++)
  {
     if(Rank!=Size-1) {
        MPI_Recv(&recv_buffer,1,MPI_DOUBLE,right_proc,BACKWARD,MPI_COMM_WORLD,&status);
     }
     else {
        recv_buffer = 0.0;
     }
     
     /* backward elimination */
     tmp_send_buffer = backward_compute(arr[i].ccof,arr[i].rhs,n,recv_buffer);
     
     /* wait & make sure that previous send command is complete */
      if(send_flag) {
         ierr = MPI_Wait(&request,&status);
      }
      send_buffer = tmp_send_buffer;
            
      /* Initiate a non-blocking send */
      ierr =MPI_Isend(&send_buffer,1,MPI_DOUBLE,left_proc,BACKWARD,MPI_COMM_WORLD,&request);
      
      send_flag = 1;
    
   }
}
  

void pll_tridiag(Tds *arr,int N,int n)
{
  /* forward elimination */
  forward(arr,N,n);
  /* backward elimination */
  backward(arr,N,n);
}  

