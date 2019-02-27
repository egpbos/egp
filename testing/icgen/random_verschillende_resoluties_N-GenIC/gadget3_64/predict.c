#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "allvars.h"
#include "proto.h"
#ifdef COSMIC_RAYS
#include "cosmic_rays.h"
#endif


void reconstruct_timebins(void)
{
  int i, n, prev, bin;
  long long glob_sum1, glob_sum2;

  for(bin = 0; bin < TIMEBINS; bin++)
    {
      TimeBinCount[bin] = 0;
      TimeBinCountSph[bin] = 0;
      FirstInTimeBin[bin] = -1;
      LastInTimeBin[bin] = -1;
#ifdef SFR
      TimeBinSfr[bin] = 0;
#endif
#ifdef BLACK_HOLES
      TimeBin_BH_mass[bin] = 0;
      TimeBin_BH_dynamicalmass[bin] = 0;
      TimeBin_BH_Mdot[bin] = 0;
      TimeBin_BH_Medd[bin] = 0;
#endif
    }

  for(i = 0; i < NumPart; i++)
    {
      bin = P[i].TimeBin;

      if(TimeBinCount[bin] > 0)
	{
	  PrevInTimeBin[i] = LastInTimeBin[bin];
	  NextInTimeBin[i] = -1;
	  NextInTimeBin[LastInTimeBin[bin]] = i;
	  LastInTimeBin[bin] = i;
	}
      else
	{
	  FirstInTimeBin[bin] = LastInTimeBin[bin] = i;
	  PrevInTimeBin[i] = NextInTimeBin[i] = -1;
	}
      TimeBinCount[bin]++;
      if(P[i].Type == 0)
	TimeBinCountSph[bin]++;

#ifdef SFR
      if(P[i].Type == 0)
	TimeBinSfr[bin] += SphP[i].Sfr;
#endif
#if BLACK_HOLES
      if(P[i].Type == 5)
	{
	  TimeBin_BH_mass[bin] += P[i].BH_Mass;
	  TimeBin_BH_dynamicalmass[bin] += P[i].Mass;
	  TimeBin_BH_Mdot[bin] += P[i].BH_Mdot;
	  TimeBin_BH_Medd[bin] += P[i].BH_Mdot / P[i].BH_Mass;
	}
#endif
    }

  FirstActiveParticle = -1;

  for(n = 0, prev = -1; n < TIMEBINS; n++)
    {
      if(TimeBinActive[n])
	for(i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i])
	  {
	    if(prev == -1)
	      FirstActiveParticle = i;

	    if(prev >= 0)
	      NextActiveParticle[prev] = i;

	    prev = i;
	  }
    }

  if(prev >= 0)
    NextActiveParticle[prev] = -1;

  sumup_large_ints(1, &NumForceUpdate, &glob_sum1);

  for(i = FirstActiveParticle, NumForceUpdate = 0; i >= 0; i = NextActiveParticle[i])
    {
      NumForceUpdate++;
      if(i >= NumPart)
	{
	  printf("Bummer i=%d\n", i);
	  endrun(12);
	}

    }

  sumup_large_ints(1, &NumForceUpdate, &glob_sum2);

  if(ThisTask == 0)
    {
      printf("sum1=%d%9d sum2=%d%9d\n",
	     (int) (glob_sum1 / 1000000000), (int) (glob_sum1 % 1000000000),
	     (int) (glob_sum2 / 1000000000), (int) (glob_sum2 % 1000000000));
    }

  if(glob_sum1 != glob_sum2 && All.NumCurrentTiStep > 0)
    endrun(121);
}



void drift_particle(int i, int time1)
{
  int j, time0, dt_step;
  double dt_drift, dt_gravkick, dt_hydrokick, dt_entr;

#ifdef DISTORTIONTENSORPS
  int j1, j2;
  /* 
   Setup all numbers that are going to be written to caustic log file. Some of them will not be set, depending
   on Makefile options. Therefore we assign zero to each value in the beginning.
  */
  MyDouble s_1=0.0, s_2=0.0, s_3=0.0;
  MyDouble smear_x=0.0, smear_y=0.0, smear_z=0.0;  
  MyDouble CurrentTurnaroundRadius=0.0;
  MyDouble caustic_counter = 0.0;
  MyDouble summed_annihilation = 0.0;
  MyDouble D_xP = 0.0; 
  MyDouble init_density = 0.0;
  MyDouble rho_normed_cutoff = 0.0;
  MyDouble second_deriv_fac = 0.0;
  MyDouble analytic_add_to_annihilation = 0.0;
  
  /* things needed for caustics detection */
  MyDouble determinant =  1.0;
  MyDouble product_matrix[3][3];
  MyDouble current_normed_stream_density;
  
  MyDouble last_normed_stream_density;

#ifdef ANNIHILATION_RADIATION
  /*1: if annihilation radiation was integrated analytically through the caustic, 0: otherwise */
  int caustic_annihilation_flag = 0;
  MyDouble rho_0, kappa_caustic;
#endif
  
#if SIM_ADAPTIVE_SOFT
  CurrentTurnaroundRadius = All.CurrentTurnaroundRadius;
#endif 
#endif /* DISTORTIONTENSORPS */

#ifdef COSMIC_RAYS
  int CRpop;
#endif

#ifdef SOFTEREQS
  double a3inv;

  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1;
#endif

  time0 = P[i].Ti_current;

  if(time1 < time0)
    {
      printf("i=%d time0=%d time1=%d\n", i, time0, time1);
      endrun(12);
    }

  if(time1 == time0)
    return;

  if(All.ComovingIntegrationOn)
    {
      dt_drift = get_drift_factor(time0, time1);
      dt_gravkick = get_gravkick_factor(time0, time1);
      dt_hydrokick = get_hydrokick_factor(time0, time1);
    }
  else
    {
      dt_drift = dt_gravkick = dt_hydrokick = (time1 - time0) * All.Timebase_interval;
    }


  for(j = 0; j < 3; j++)
    P[i].Pos[j] += P[i].Vel[j] * dt_drift;


/* START PHASE-SPACE ANALYSIS ------------------------------------------------------ */
#ifdef DISTORTIONTENSORPS

  /* do the DRIFT for distortion (this updates D_xq and D_xp; D_vq and D_vp are updated during kick) */
  for(j1 = 0; j1 < 3; j1++)
    for(j2 = 0; j2 < 6; j2++)
      P[i].distortion_tensorps[j1][j2] += P[i].distortion_tensorps[j1 + 3][j2] * dt_drift;

#ifdef REINIT_AT_TURNAROUND
  /* do we have to reinit distortion tensor? Only particles that are initially outside the turnaround radius are analyzed */
  if ((P[i].Pos[0]*P[i].Vel[0]+P[i].Pos[1]*P[i].Vel[1]+P[i].Pos[2]*P[i].Vel[2] < 0.0) && (P[i].turnaround_flag == 0))
    {
      /* yes, particle turned around */
      /* re-init distortion tensor */
      for(j1 = 0; j1 < 6; j1++)
	for(j2 = 0; j2 < 6; j2++)
	  {
	    if((j1 == j2))
	      {
		P[i].distortion_tensorps[j1][j2] = 1.0;
	      }
	    else
	      {
		P[i].distortion_tensorps[j1][j2] = 0.0;
	      }
	  }
      /* mark particle so that we know that it already turned around */    
      P[i].turnaround_flag = 1;          
      
      /* set new initial sheet orientation since we start distortion from this point (analytic solution) */       
      for (j1=0; j1<=2; j1++)
        for (j2=0; j2<=2; j2++)
          P[i].V_matrix[j1][j2] = P[i].Pos[j1]*P[i].Pos[j2] / (P[i].Pos[0]*P[i].Pos[0]+P[i].Pos[1]*P[i].Pos[1]+P[i].Pos[2]*P[i].Pos[2]) *
                                  (3.0/4.0*M_PI)*(3.0/4.0*M_PI) / (3.0 + 1.0/All.SIM_epsilon) * 1.0 / All.Time;
      /* 
       The correct stream init density at turn around is given by rho_crit * 9.0 * M_PI * M_PI / (16.0*(3.0*epsilon+1.0)).
       This is exactly the density at the turnaround radius for a given density perturbation index epsilon.
       (see for example, Sikivie et al 1997) 
      */
      P[i].init_density = 9.0 * M_PI * M_PI / (16.0*(3.0*All.SIM_epsilon+1.0)) * 1.0 / (6.0 * M_PI * All.G * All.Time * All.Time);                 

      /* reset last_stream_determinant (needed for caustic identification) */
      P[i].last_stream_determinant = 1.0;
#ifdef ANNIHILATION_RADIATION  
      /* set annihilation radiation to zero */
      P[i].annihilation = 0.0;
#endif          
    }    
#endif

  /* NOW CAUSTIC AND ANNIHILATION HANDLING */
#ifdef REINIT_AT_TURNAROUND
 /* only analyse particle initially outside the turnaround radius, those initially inside have been set to P[i].turnaround_flag=-1 */
 if (P[i].turnaround_flag >= 0) 
 {
#endif  
  /* CAUSTIC IDENTIFICATION PART */
  /* projection phase-space distrtion matrix (see Vogelsberger et al, 2008 Eq. (22)) -> needed to check for caustic in drift */
  product_matrix[0][0] = P[i].distortion_tensorps[0][0] + 
                         P[i].distortion_tensorps[0][3] * P[i].V_matrix[0][0] +
                         P[i].distortion_tensorps[0][4] * P[i].V_matrix[1][0] +
                         P[i].distortion_tensorps[0][5] * P[i].V_matrix[2][0];
  product_matrix[0][1] = P[i].distortion_tensorps[0][1] + 
                         P[i].distortion_tensorps[0][3] * P[i].V_matrix[0][1] +
                         P[i].distortion_tensorps[0][4] * P[i].V_matrix[1][1] +
                         P[i].distortion_tensorps[0][5] * P[i].V_matrix[2][1];
  product_matrix[0][2] = P[i].distortion_tensorps[0][2] + 
                         P[i].distortion_tensorps[0][3] * P[i].V_matrix[0][2] +
                         P[i].distortion_tensorps[0][4] * P[i].V_matrix[1][2] +
                         P[i].distortion_tensorps[0][5] * P[i].V_matrix[2][2];
  product_matrix[1][0] = P[i].distortion_tensorps[1][0] + 
                         P[i].distortion_tensorps[1][3] * P[i].V_matrix[0][0] +
                         P[i].distortion_tensorps[1][4] * P[i].V_matrix[1][0] +
                         P[i].distortion_tensorps[1][5] * P[i].V_matrix[2][0];
  product_matrix[1][1] = P[i].distortion_tensorps[1][1] + 
                         P[i].distortion_tensorps[1][3] * P[i].V_matrix[0][1] +
                         P[i].distortion_tensorps[1][4] * P[i].V_matrix[1][1] +
                         P[i].distortion_tensorps[1][5] * P[i].V_matrix[2][1];
  product_matrix[1][2] = P[i].distortion_tensorps[1][2] + 
                         P[i].distortion_tensorps[1][3] * P[i].V_matrix[0][2] +
                         P[i].distortion_tensorps[1][4] * P[i].V_matrix[1][2] +
                         P[i].distortion_tensorps[1][5] * P[i].V_matrix[2][2];
  product_matrix[2][0] = P[i].distortion_tensorps[2][0] + 
                         P[i].distortion_tensorps[2][3] * P[i].V_matrix[0][0] +
                         P[i].distortion_tensorps[2][4] * P[i].V_matrix[1][0] +
                         P[i].distortion_tensorps[2][5] * P[i].V_matrix[2][0];
  product_matrix[2][1] = P[i].distortion_tensorps[2][1] +  
                         P[i].distortion_tensorps[2][3] * P[i].V_matrix[0][1] +
                         P[i].distortion_tensorps[2][4] * P[i].V_matrix[1][1] +
                         P[i].distortion_tensorps[2][5] * P[i].V_matrix[2][1];
  product_matrix[2][2] = P[i].distortion_tensorps[2][2] + 
                         P[i].distortion_tensorps[2][3] * P[i].V_matrix[0][2] +
                         P[i].distortion_tensorps[2][4] * P[i].V_matrix[1][2] +
                         P[i].distortion_tensorps[2][5] * P[i].V_matrix[2][2];

  /* this determinant will change sign when we pass through a caustic -> criterion for caustics */
  determinant = ((product_matrix[0][0]) * (product_matrix[1][1]) * (product_matrix[2][2]) +
		 (product_matrix[0][1]) * (product_matrix[1][2]) * (product_matrix[2][0]) +
		 (product_matrix[0][2]) * (product_matrix[1][0]) * (product_matrix[2][1]) -
		 (product_matrix[0][2]) * (product_matrix[1][1]) * (product_matrix[2][0]) -
		 (product_matrix[0][0]) * (product_matrix[1][2]) * (product_matrix[2][1]) -
		 (product_matrix[0][1]) * (product_matrix[1][0]) * (product_matrix[2][2]));

  /* 
    Current and last NORMED stream density, linear order result of the last and current timestep. 
  */  
  current_normed_stream_density = 1.0/fabs(determinant);
  last_normed_stream_density    = 1.0/fabs(P[i].last_stream_determinant);  
  
#ifdef ANNIHILATION_RADIATION  
  /* avarage stream density */
  rho_0 = (current_normed_stream_density + last_normed_stream_density) / 2;

  /* extract phase-space information for annihilation radiation calculation -> cutoff density */
  P[i].rho_normed_cutoff_last    = P[i].rho_normed_cutoff_current;
  P[i].rho_normed_cutoff_current = analyse_phase_space(i, &s_1, &s_2, &s_3, &smear_x, &smear_y, &smear_z, &D_xP, &second_deriv_fac);
#endif

  /* CAUSTIC FOUND? -> was there a caustics between the last and the current timestep and does the particle type fit? */
  if((determinant * P[i].last_stream_determinant < 0.0) && (((1 << P[i].Type) & (CAUSTIC_FINDER))))
    {
 
#ifdef ANNIHILATION_RADIATION
      /* get normed_cutoff density */
      rho_normed_cutoff = P[i].rho_normed_cutoff_current;
      
      /* 
       If rho_0 = (current_normed_stream_density + last_normed_stream_density) / 2 (so the average of current and last
       normed stream density) is below current normed cutoff density, we integrate analytically through the caustic
       for that time step. 

       rho  
       ^
       |         *  <--- rho_normed_cutoff
       |        ***
       |       *****        
       |      *******   <-- area under curve ~ diverges logarithmicly 
       |     ********* 
       |    *********** 
       |   *************  <--- rho_0 
       |---|-----|-----|----> t
         t_last t_c   t_current 
                    

       Note that the stream density is assumed to be symmetric around the caustic point: t_caustic = (t_last + t_current) / 2 

       So we can center everything around t=0:

       rho  
       ^
       |         *  <--- rho_normed_cutoff
       |        ***
       |       *****        
       |      *******   <-- area under curve ~ diverges logarithmicly 
       |     ********* 
       |    *********** 
       |   *************  <--- rho_0 
       |---|-----|-----|----> t
    -dt_drift/2  0   +dt_drift/2

       t_caustic = 0  and  dt_drift = t_current - t_last        
       
       The fact that the numerical normed density near the caustic is below the cutoff, means that we cannot
       numerically resolve the very high caustic density within the time step. 
       
       The analytic integration exploits the logarithmic divergence:
       
       Ansatz:
       rho(t) = kappa / |t|   --->   rho_0 = kappa / |dt_drift/2|   --->  kappa = dt_drift/2 * rho_0
       
       Integrated annihilation rate:
       
       2 \int_{-dt_drift/2}^{+dt_drift/2} dt rho(t) = 
       2 - 2 kappa \int_{-dt_drift/2}^{0} dt 1/t  =  [ substitute drho / dt = + kappa/t^2 for t <0]
       2 2 kappa \int_{rho_0}^{rho_c} drho 1/rho =
       2 2 kappa ln(rho_c/rho_0)
       
       Therefore we need to add 2 kappa ln(\rho_c/\rho_0) when passing through an unresolved caustic,
       where rho_c is the calculated maximum density in the caustic.
              
       Note:
       We assume that the maximum density for the current and last time step do not differ much, so
       that we can compare rho_0 always to the current maximum density.
       
      */ 
#ifdef REINIT_AT_TURNAROUND    
      /* 
       We only integrate annihilation for particles that turned around already (otherwise wrong initial density) 
       This is a security check, since each particle going through a caustic in general already turned around before.
      */
      if (P[i].turnaround_flag == 1) 
      {
#endif 
      /* if this condition is true we have to integrate analyitcally through the caustic */        
      if (rho_0 < P[i].rho_normed_cutoff_current)
       {
        /* we do have an analytic integration through the caustic -> increase the counter and mark the flag */
        P[i].analytic_caustics          += 1.0;
        caustic_annihilation_flag        = 1;
        /* calculate analytic contribution */ 
        kappa_caustic                    = dt_drift / 2.0 * rho_0;
        analytic_add_to_annihilation     = 2.0 * kappa_caustic * log(fabs(P[i].rho_normed_cutoff_current/rho_0));

        /* add contribution to annihilation radiation */
        P[i].annihilation += analytic_add_to_annihilation;
       }        
      summed_annihilation  = P[i].annihilation;
#ifdef REINIT_AT_TURNAROUND
      } /* end if (P[i].turnaround_flag == 1) */
#endif  

#endif /* ANNIHILATION_RADIATION */

      /* increase caustic counter by one */
      P[i].caustic_counter += 1.0;
      caustic_counter       = P[i].caustic_counter;

#ifdef OUTPUT_LAST_CAUSTIC       
      P[i].lc_Time    = All.Time;
      P[i].lc_Pos[0]  = P[i].Pos[0];
      P[i].lc_Pos[1]  = P[i].Pos[1];
      P[i].lc_Pos[2]  = P[i].Pos[2];
      P[i].lc_Vel[0]  = P[i].Vel[0];
      P[i].lc_Vel[1]  = P[i].Vel[1];
      P[i].lc_Vel[2]  = P[i].Vel[2];
#ifdef ANNIHILATION_RADIATION      
      P[i].lc_rho_normed_cutoff = P[i].rho_normed_cutoff_current;
#else
      P[i].lc_rho_normed_cutoff = 0.0;
#endif
#endif

      init_density = P[i].init_density;
      
      /* write data to caustic log file -> allows caustics to be tracked on time-stepping frequency */
      fprintf(FdCaustics, "%g %g %d %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",
	      (MyOutputFloat) All.Time, 
              (MyOutputFloat) determinant, 
               P[i].ID,
	      (MyOutputFloat) P[i].Pos[0], (MyOutputFloat) P[i].Pos[1], (MyOutputFloat) P[i].Pos[2],
	      (MyOutputFloat) P[i].Vel[0], (MyOutputFloat) P[i].Vel[1], (MyOutputFloat) P[i].Vel[2],
              caustic_counter,
              s_1, s_2, s_3, smear_x, smear_y, smear_z, 
              summed_annihilation,
              CurrentTurnaroundRadius,
              D_xP,
	      init_density / (1.0/(6.0*M_PI*All.G*All.Time*All.Time)),
              rho_normed_cutoff,
              second_deriv_fac,
              analytic_add_to_annihilation);
      fflush(FdCaustics);         
    }

#ifdef ANNIHILATION_RADIATION
  /* is the numerical stream density above the normed_cutoff -> cut it */
  if (rho_0 > P[i].rho_normed_cutoff_current)  
   {
    rho_0 = P[i].rho_normed_cutoff_current;
   }
  
  /* save cutted averaged stream density in physical units -> multiply with initial stream density */
  P[i].stream_density = P[i].init_density * rho_0;
#ifdef REINIT_AT_TURNAROUND    
  /* 
    We only integrate annihilation for particles that turned around already (otherwise wrong initial density) 
    This is a security check, since each particle going through a caustic in general already turned around before.
  */
  if (P[i].turnaround_flag == 1) 
   {
#endif  
    /* integrate normed annihilation radiation -> multiplication with initial stream density is done in io.c to reduce round-off */
    if (caustic_annihilation_flag == 0)
      P[i].annihilation += rho_0 * dt_drift;      
#ifdef REINIT_AT_TURNAROUND
   } /* end if (P[i].turnaround_flag == 1) */
#endif  

#ifdef NO_CENTER_ANNIHILATION
  /* ignore the center for annihilation */
  if (sqrt(P[i].Pos[0]*P[i].Pos[0] + P[i].Pos[1]*P[i].Pos[1] + P[i].Pos[2]*P[i].Pos[2]) < All.SofteningTable[P[i].Type]) 
   {
    /* set everything to zero in center */
    P[i].annihilation   = 0.0;
    P[i].stream_density = 0.0;
   }
#endif
#endif /*ANNIHILATION_RADIATION */

#ifdef REINIT_AT_TURNAROUND
 } /* end if (P[i].turnaround_flag >= 0) */
#endif  

 /* update determinant, so we can identify the next caustic along the orbit */
 P[i].last_stream_determinant = determinant;

#endif /* DISTORTIONTENSORPS */
/* END PHASE-SPACE ANALYSIS ------------------------------------------------------ */


#ifndef HPM
  if(P[i].Type == 0)
    {
#ifdef PMGRID
      for(j = 0; j < 3; j++)
	SphP[i].VelPred[j] +=
	  (P[i].g.GravAccel[j] + P[i].GravPM[j]) * dt_gravkick + SphP[i].a.HydroAccel[j] * dt_hydrokick;
#else
      for(j = 0; j < 3; j++)
	SphP[i].VelPred[j] += P[i].g.GravAccel[j] * dt_gravkick + SphP[i].a.HydroAccel[j] * dt_hydrokick;
#endif

      SphP[i].d.Density *= exp(-SphP[i].v.DivVel * dt_drift);
      PPP[i].Hsml *= exp(0.333333333333 * SphP[i].v.DivVel * dt_drift);

      if(PPP[i].Hsml < All.MinGasHsml)
	PPP[i].Hsml = All.MinGasHsml;

#ifndef WAKEUP
      dt_step = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0);
#else
      dt_step = P[i].dt_step;
#endif
      dt_entr = (time1 - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;

#ifndef EOS_DEGENERATE
#ifndef MHM
#ifndef SOFTEREQS
#ifndef VORONOI_MESHRELAX

      SphP[i].Pressure = (SphP[i].Entropy + SphP[i].e.DtEntropy * dt_entr) * pow(SphP[i].d.Density, GAMMA);

#endif
#else
      if(SphP[i].d.Density * a3inv >= All.PhysDensThresh)
	SphP[i].Pressure =
	  All.FactorForSofterEQS * (SphP[i].Entropy +
				    SphP[i].e.DtEntropy * dt_entr) * pow(SphP[i].d.Density,
									 GAMMA) + (1 -
										   All.
										   FactorForSofterEQS) *
	  GAMMA_MINUS1 * SphP[i].d.Density * All.InitGasU;
      else
	SphP[i].Pressure = (SphP[i].Entropy + SphP[i].e.DtEntropy * dt_entr) * pow(SphP[i].d.Density, GAMMA);
#endif
#else
      /* Here we use an isothermal equation of state */
      SphP[i].Pressure = GAMMA_MINUS1 * SphP[i].d.Density * All.InitGasU;
      SphP[i].Entropy = SphP[i].Pressure / pow(SphP[i].d.Density, GAMMA);
#endif
#else
      /* call tabulated eos with physical units */
#ifdef EOS_ENERGY
      eos_calc_egiven2(SphP[i].d.Density * All.UnitDensity_in_cgs, SphP[i].xnuc,
		       SphP[i].Entropy + SphP[i].e.DtEntropy * dt_entr * All.UnitTime_in_s, &SphP[i].temp,
		       &SphP[i].Pressure);
      SphP[i].Pressure /= All.UnitPressure_in_cgs;
#else
      eos_calc_sgiven(SphP[i].d.Density * All.UnitDensity_in_cgs, SphP[i].xnuc,
		      SphP[i].Entropy + SphP[i].e.DtEntropy * dt_entr * All.UnitTime_in_s, &SphP[i].temp,
		      &SphP[i].Pressure, &SphP[i].u);
      SphP[i].Pressure /= All.UnitPressure_in_cgs;
#endif
#endif

#ifdef COSMIC_RAYS
#if defined( CR_UPDATE_PARANOIA )
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	CR_Particle_Update(SphP + i, CRpop);
#endif
#ifndef CR_NOPRESSURE
      for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	SphP[i].Pressure += CR_Comoving_Pressure(SphP + i, CRpop);
#endif
#endif


#if defined(MAGNETIC) && !defined(EULERPOTENTIALS)
      for(j = 0; j < 3; j++)
	SphP[i].BPred[j] += SphP[i].DtB[j] * dt_entr;
#ifdef DIVBCLEANING_DEDNER
      SphP[i].PhiPred += SphP[i].DtPhi * dt_entr;
#endif
#endif

    }
#endif /* end of HPM */

  P[i].Ti_current = time1;
}



void move_particles(int time1)
{
  int i;

  if(ThisTask == 0)
    printf("MOVE\n");

  for(i = 0; i < NumPart; i++)
    drift_particle(i, time1);
}



/*! This function makes sure that all particle coordinates (Pos) are
 *  periodically mapped onto the interval [0, BoxSize].  After this function
 *  has been called, a new domain decomposition should be done, which will
 *  also force a new tree construction.
 */
#ifdef PERIODIC
void do_box_wrapping(void)
{
  int i, j;
  double boxsize[3];

  for(j = 0; j < 3; j++)
    boxsize[j] = All.BoxSize;

#ifdef LONG_X
  boxsize[0] *= LONG_X;
#endif
#ifdef LONG_Y
  boxsize[1] *= LONG_Y;
#endif
#ifdef LONG_Z
  boxsize[2] *= LONG_Z;
#endif

  for(i = 0; i < NumPart; i++)
    for(j = 0; j < 3; j++)
      {
	while(P[i].Pos[j] < 0)
	  P[i].Pos[j] += boxsize[j];

	while(P[i].Pos[j] >= boxsize[j])
	  P[i].Pos[j] -= boxsize[j];
      }
}
#endif



/*

#ifdef XXLINFO
#ifdef MAGNETIC
  double MeanB_part = 0, MeanB_sum;

#ifdef TRACEDIVB
  double MaxDivB_part = 0, MaxDivB_all;
  double dmax1, dmax2;
#endif
#endif
#ifdef TIME_DEP_ART_VISC
  double MeanAlpha_part = 0, MeanAlpha_sum;
#endif
#endif



#ifdef XXLINFO
      if(Flag_FullStep == 1)
        {
#ifdef MAGNETIC
          MeanB_part += sqrt(SphP[i].BPred[0] * SphP[i].BPred[0] +
                             SphP[i].BPred[1] * SphP[i].BPred[1] + SphP[i].BPred[2] * SphP[i].BPred[2]);
#ifdef TRACEDIVB
          MaxDivB_part = DMAX(MaxDivB, fabs(SphP[i].divB));
#endif
#endif
#ifdef TIME_DEP_ART_VISC
          MeanAlpha_part += SphP[i].alpha;
#endif
        }
#endif


#ifdef XXLINFO
  if(Flag_FullStep == 1)
    {
#ifdef MAGNETIC
      MPI_Reduce(&MeanB_part, &MeanB_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if(ThisTask == 0)
	MeanB = MeanB_sum / All.TotN_gas;
#ifdef TRACEDIVB
      MPI_Reduce(&MaxDivB_part, &MaxDivB_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if(ThisTask == 0)
	MaxDivB = MaxDivB_all;
#endif
#endif
#ifdef TIME_DEP_ART_VISC
      MPI_Reduce(&MeanAlpha_part, &MeanAlpha_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if(ThisTask == 0)
	MeanAlpha = MeanAlpha_sum / All.TotN_gas;
#endif
    }
#endif
*/
