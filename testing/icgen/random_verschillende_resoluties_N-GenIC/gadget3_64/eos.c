#ifdef EOS_DEGENERATE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"
#include "eos.h"

void eos_init()
{
  int swap, skip;
  FILE *fd;
  int i, entries, bytes;

  fd = 0;
  if(ThisTask == 0)
    {
      eos_checkswap(All.EosTable, &swap);

      if(swap == 2)
	{
	  printf("eos table `%s' is corrupt.\n", All.EosTable);
	  endrun(1337);
	}

      if(!(fd = fopen(All.EosTable, "r")))
	{
	  printf("can't open file `%s' for reading eos table.\n", All.EosTable);
	  endrun(1337);
	}

      fread(&skip, sizeof(int), 1, fd);
      fread(&eos_table.ntemp, sizeof(int), 1, fd);
      fread(&eos_table.nrho, sizeof(int), 1, fd);
      fread(&eos_table.nye, sizeof(int), 1, fd);
      fread(&skip, sizeof(int), 1, fd);

      fread(&skip, sizeof(int), 1, fd);
      fread(&eos_table.ltempMin, sizeof(double), 1, fd);
      fread(&eos_table.ltempMax, sizeof(double), 1, fd);
      fread(&eos_table.lrhoMin, sizeof(double), 1, fd);
      fread(&eos_table.lrhoMax, sizeof(double), 1, fd);
      fread(&eos_table.yeMin, sizeof(double), 1, fd);
      fread(&eos_table.yeMax, sizeof(double), 1, fd);
      fread(&skip, sizeof(int), 1, fd);

      if(swap)
	{
	  eos_table.ntemp = eos_SwapInt(eos_table.ntemp);
	  eos_table.nrho = eos_SwapInt(eos_table.nrho);
	  eos_table.nye = eos_SwapInt(eos_table.nye);

	  eos_table.ltempMin = eos_SwapDouble(eos_table.ltempMin);
	  eos_table.ltempMax = eos_SwapDouble(eos_table.ltempMax);
	  eos_table.lrhoMin = eos_SwapDouble(eos_table.lrhoMin);
	  eos_table.lrhoMax = eos_SwapDouble(eos_table.lrhoMax);
	  eos_table.yeMin = eos_SwapDouble(eos_table.yeMin);
	  eos_table.yeMax = eos_SwapDouble(eos_table.yeMax);
	}
    }

  MPI_Bcast(&eos_table.ntemp, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eos_table.nrho, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eos_table.nye, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&eos_table.ltempMin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eos_table.ltempMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eos_table.lrhoMin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eos_table.lrhoMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eos_table.yeMin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eos_table.yeMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  eos_table.ltempDelta = (eos_table.ltempMax - eos_table.ltempMin) / (eos_table.ntemp - 1);
  eos_table.lrhoDelta = (eos_table.lrhoMax - eos_table.lrhoMin) / (eos_table.nrho - 1);
  eos_table.yeDelta = (eos_table.yeMax - eos_table.yeMin) / (eos_table.nye - 1);

  eos_table.tempMin = pow(10.0, eos_table.ltempMin);
  eos_table.tempMax = pow(10.0, eos_table.ltempMax);
  eos_table.rhoMin = pow(10.0, eos_table.lrhoMin);
  eos_table.rhoMax = pow(10.0, eos_table.lrhoMax);

  if(ThisTask == 0)
    {
      printf("EOS table spans rho [%e,%e], temp [%e,%e], ye [%e,%e].\n", eos_table.rhoMin, eos_table.rhoMax,
	     eos_table.tempMin, eos_table.tempMax, eos_table.yeMin, eos_table.yeMax);
      printf("Resolution is rho: %d, temp: %d, ye: %d points.\n", eos_table.nrho, eos_table.ntemp,
	     eos_table.nye);
    }

  eos_table.ltemp = (double *) malloc(eos_table.ntemp * sizeof(double));
  eos_table.lrho = (double *) malloc(eos_table.nrho * sizeof(double));
  eos_table.ye = (double *) malloc(eos_table.nye * sizeof(double));

  if(!(eos_table.ltemp && eos_table.lrho && eos_table.ye))
    {
      printf("not enough memory to allocate eos arrays.\n");
      endrun(1337);
    }

  eos_table.ltemp[0] = eos_table.ltempMin;
  for(i = 1; i < eos_table.ntemp; i++)
    eos_table.ltemp[i] = eos_table.ltemp[i - 1] + eos_table.ltempDelta;
  eos_table.lrho[0] = eos_table.lrhoMin;
  for(i = 1; i < eos_table.nrho; i++)
    eos_table.lrho[i] = eos_table.lrho[i - 1] + eos_table.lrhoDelta;
  eos_table.ye[0] = eos_table.yeMin;
  for(i = 1; i < eos_table.nye; i++)
    eos_table.ye[i] = eos_table.ye[i - 1] + eos_table.yeDelta;

  eos_table.ltempDeltaI = 1. / eos_table.ltempDelta;
  eos_table.lrhoDeltaI = 1. / eos_table.lrhoDelta;
  eos_table.yeDeltaI = 1. / eos_table.yeDelta;

  entries = eos_table.ntemp * eos_table.nrho * eos_table.nye;
  bytes = entries * sizeof(double);
  eos_table.p = (double *) malloc(bytes);
  eos_table.e = (double *) malloc(bytes);
  eos_table.dedt = (double *) malloc(bytes);
#ifndef EOS_ENERGY
  eos_table.s = (double *) malloc(bytes);
  eos_table.dsdt = (double *) malloc(bytes);
#endif

  if(!(eos_table.p && eos_table.e && eos_table.dedt))
    {
      printf("not enough memory to allocate eos arrays.\n");
      endrun(1337);
    }
#ifndef EOS_ENERGY
  if(!(eos_table.s && eos_table.dsdt))
    {
      printf("not enough memory to allocate eos arrays.\n");
      endrun(1337);
    }
#endif

  if(ThisTask == 0)
    {
      printf("Reading grid containing %d points for each quantity.\n", entries);

      fread(&skip, sizeof(int), 1, fd);
      fread(eos_table.p, sizeof(double), entries, fd);
      fseek(fd, 2 * sizeof(double) * entries, SEEK_CUR);	// skip dpdt and dpdr
      fread(eos_table.e, sizeof(double), entries, fd);
      fread(eos_table.dedt, sizeof(double), entries, fd);

#ifndef EOS_ENERGY
      fseek(fd, 2 * sizeof(double) * entries, SEEK_CUR);	// skip dedr
      fread(eos_table.s, sizeof(double), entries, fd);
      fread(eos_table.dsdt, sizeof(double), entries, fd);
#endif
      fread(&skip, sizeof(int), 1, fd);

      if(swap)
	{
	  for(i = 0; i < entries; i++)
	    {
	      eos_table.p[i] = eos_SwapDouble(eos_table.p[i]);
	      eos_table.e[i] = eos_SwapDouble(eos_table.e[i]);
	      eos_table.dedt[i] = eos_SwapDouble(eos_table.dedt[i]);
#ifndef EOS_ENERGY
	      eos_table.s[i] = eos_SwapDouble(eos_table.s[i]);
	      eos_table.dsdt[i] = eos_SwapDouble(eos_table.dsdt[i]);
#endif
	    }
	}

      fclose(fd);
    }

  MPI_Bcast(eos_table.p, entries, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(eos_table.e, entries, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(eos_table.dedt, entries, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#ifndef EOS_ENERGY
  MPI_Bcast(eos_table.s, entries, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(eos_table.dsdt, entries, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  eos_table.nuclearmasses = (double *) malloc(EOS_NSPECIES * sizeof(double));
  eos_table.nuclearmasses[0] = 1.0;
  eos_table.nuclearmasses[1] = 4.0;
  eos_table.nuclearmasses[2] = 12.0;

  eos_table.nuclearcharges = (double *) malloc(EOS_NSPECIES * sizeof(double));
  eos_table.nuclearcharges[0] = 1.0;
  eos_table.nuclearcharges[1] = 2.0;
  eos_table.nuclearcharges[2] = 6.0;

  if(ThisTask == 0)
    {
      printf("EOS init done.\n");
    }
}

void eos_deinit()
{
  free(eos_table.p);
  free(eos_table.e);
  free(eos_table.dedt);
#ifndef EOS_ENERGY
  free(eos_table.s);
  free(eos_table.dsdt);
#endif
  free(eos_table.nuclearmasses);
}

void eos_calc_egiven2(double rho, double *xnuc, double e, double *temp, double *p)
{
  double *n, ni, nb, ne;	// number densities [ particles per volume ]
  double ye;			// electron fraction ( electrons per baryon )
  double _temp, _tempold;	// temperatures [ K ]
  double _ee, _deedt;		// energy per mass of electrons + positrons + radiation 
  double _ie, _diedt;		// energy per mass of ions
  double _e, _ep;		// total energy per mass, pressure of electrons
  int iter, i;

  if(*temp == 0)
    return;

  if(rho > eos_table.rhoMax)
    {
      printf("Density exceeds allowed maximum, rho: %g, maximum: %g\n", rho, eos_table.rhoMax);
      endrun(1337);
      return;
    }

  n = (double *) malloc(EOS_NSPECIES * sizeof(double));
  ni = 0.0;
  ne = 0.0;
  for(i = 0; i < EOS_NSPECIES; i++)
    {
      n[i] = xnuc[i] * rho * AVOGADRO / eos_table.nuclearmasses[i];
      ni += n[i];
      ne += n[i] * eos_table.nuclearcharges[i];
    }

  if(rho <= eos_table.rhoMin)
    {
      // assume ideal fully ionised gas
      *temp = 2.0 / 3.0 * e * rho / (ni + ne) / BOLTZMANN;
      *p = (ni + ne) * BOLTZMANN * (*temp);
      return;
    }

  nb = rho * AVOGADRO;
  ye = ne / nb;

  if(ye < eos_table.yeMin || ye > eos_table.yeMax)
    {
      printf("Electron fraction out of table, ye: %g, table: [%g,%g]\n", ye, eos_table.yeMin,
	     eos_table.yeMax);
      endrun(1337);
      return;
    }

  _diedt = 1.5 * ni * BOLTZMANN / rho;

  if(*temp == 1.0)
    {
      // lets make a guess for the temperatures, assuming an ideal gas
      _temp = 2.0 / 3.0 * e * rho / (ni + ne) / BOLTZMANN;
    }
  else
    {
      _temp = *temp;
    }

  _tempold = 0.0;
  iter = 0;
  while(iter < EOS_MAXITER)
    {
      eos_trilinear_e(_temp, rho, ye, &_ee, &_deedt);

      _ie = _diedt * _temp;
      _e = _ee + _ie;
      if(fabs(_e - e) <= (EOS_EPS * e))
	{
	  break;
	}

      _tempold = _temp;
      _temp = _temp - (_e - e) / (_deedt + _diedt);
      if(_temp <= eos_table.tempMin && _tempold == eos_table.tempMin)
	{
	  _temp = eos_table.tempMin;
	  break;
	}

      if(_temp >= eos_table.tempMax && _tempold >= eos_table.tempMax)
	{
	  _temp = eos_table.tempMax;
	  break;
	}

      _temp = _temp < eos_table.tempMax ? _temp : eos_table.tempMax;
      _temp = _temp > eos_table.tempMin ? _temp : eos_table.tempMin;
      iter++;
    }

  eos_trilinear2(_temp, rho, ye, &_ee, &_ep);

  *temp = _temp;
  *p = ni * BOLTZMANN * (*temp) + _ep;

  free(n);
}

void eos_trilinear_e(double temp, double rho, double ye, double *e, double *dedt)
{
  double logtemp, logrho;
  int itemp, irho, iye;
  unsigned int entry111, entry211, entry121, entry221, entry112, entry212, entry122, entry222;
  double w111, w211, w121, w221, w112, w212, w122, w222;
  double dx1, dx2, dy1, dy2, dz1, dz2;

  logtemp = log10(temp);
  logrho = log10(rho);

  itemp = (logtemp - eos_table.ltempMin) / eos_table.ltempDelta;
  irho = (logrho - eos_table.lrhoMin) / eos_table.lrhoDelta;
  iye = (ye - eos_table.yeMin) / eos_table.yeDelta;

  dx1 = (logtemp - eos_table.ltemp[itemp]) * eos_table.ltempDeltaI;
  dy1 = (logrho - eos_table.lrho[irho]) * eos_table.lrhoDeltaI;
  dz1 = (ye - eos_table.ye[iye]) * eos_table.yeDeltaI;
  dx2 = 1. - dx1;
  dy2 = 1. - dy1;
  dz2 = 1. - dz1;

  w111 = dx2 * dy2 * dz2;
  w112 = dx2 * dy2 * dz1;
  w121 = dx2 * dy1 * dz2;
  w122 = dx2 * dy1 * dz1;
  w211 = dx1 * dy2 * dz2;
  w212 = dx1 * dy2 * dz1;
  w221 = dx1 * dy1 * dz2;
  w222 = dx1 * dy1 * dz1;

  entry111 = (iye * eos_table.nrho + irho) * eos_table.ntemp + itemp;
  entry112 = ((iye + 1) * eos_table.nrho + irho) * eos_table.ntemp + itemp;
  entry121 = (iye * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp;
  entry122 = ((iye + 1) * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp;
  entry211 = (iye * eos_table.nrho + irho) * eos_table.ntemp + itemp + 1;
  entry212 = ((iye + 1) * eos_table.nrho + irho) * eos_table.ntemp + itemp + 1;
  entry221 = (iye * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp + 1;
  entry222 = ((iye + 1) * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp + 1;

  *e = eos_table.e[entry111] * w111 + eos_table.e[entry112] * w112 +
    eos_table.e[entry121] * w121 + eos_table.e[entry122] * w122 +
    eos_table.e[entry211] * w211 + eos_table.e[entry212] * w212 +
    eos_table.e[entry221] * w221 + eos_table.e[entry222] * w222;

  *dedt = eos_table.dedt[entry111] * w111 + eos_table.dedt[entry112] * w112 +
    eos_table.dedt[entry121] * w121 + eos_table.dedt[entry122] * w122 +
    eos_table.dedt[entry211] * w211 + eos_table.dedt[entry212] * w212 +
    eos_table.dedt[entry221] * w221 + eos_table.dedt[entry222] * w222;
}

void eos_trilinear2(double temp, double rho, double ye, double *e, double *p)
{
  double logtemp, logrho;
  int itemp, irho, iye;
  unsigned int entry111, entry211, entry121, entry221, entry112, entry212, entry122, entry222;
  double w111, w211, w121, w221, w112, w212, w122, w222;
  double dx1, dx2, dy1, dy2, dz1, dz2;

  logtemp = log10(temp);
  logrho = log10(rho);

  itemp = (logtemp - eos_table.ltempMin) / eos_table.ltempDelta;
  irho = (logrho - eos_table.lrhoMin) / eos_table.lrhoDelta;
  iye = (ye - eos_table.yeMin) / eos_table.yeDelta;

  dx1 = (logtemp - eos_table.ltemp[itemp]) * eos_table.ltempDeltaI;
  dy1 = (logrho - eos_table.lrho[irho]) * eos_table.lrhoDeltaI;
  dz1 = (ye - eos_table.ye[iye]) * eos_table.yeDeltaI;
  dx2 = 1. - dx1;
  dy2 = 1. - dy1;
  dz2 = 1. - dz1;

  w111 = dx2 * dy2 * dz2;
  w112 = dx2 * dy2 * dz1;
  w121 = dx2 * dy1 * dz2;
  w122 = dx2 * dy1 * dz1;
  w211 = dx1 * dy2 * dz2;
  w212 = dx1 * dy2 * dz1;
  w221 = dx1 * dy1 * dz2;
  w222 = dx1 * dy1 * dz1;

  entry111 = (iye * eos_table.nrho + irho) * eos_table.ntemp + itemp;
  entry112 = ((iye + 1) * eos_table.nrho + irho) * eos_table.ntemp + itemp;
  entry121 = (iye * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp;
  entry122 = ((iye + 1) * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp;
  entry211 = (iye * eos_table.nrho + irho) * eos_table.ntemp + itemp + 1;
  entry212 = ((iye + 1) * eos_table.nrho + irho) * eos_table.ntemp + itemp + 1;
  entry221 = (iye * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp + 1;
  entry222 = ((iye + 1) * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp + 1;

  *e = eos_table.e[entry111] * w111 + eos_table.e[entry112] * w112 +
    eos_table.e[entry121] * w121 + eos_table.e[entry122] * w122 +
    eos_table.e[entry211] * w211 + eos_table.e[entry212] * w212 +
    eos_table.e[entry221] * w221 + eos_table.e[entry222] * w222;

  *p = eos_table.p[entry111] * w111 + eos_table.p[entry112] * w112 +
    eos_table.p[entry121] * w121 + eos_table.p[entry122] * w122 +
    eos_table.p[entry211] * w211 + eos_table.p[entry212] * w212 +
    eos_table.p[entry221] * w221 + eos_table.p[entry222] * w222;
}

#ifndef EOS_ENERGY
void eos_calc_dsdt(double temp, double rho, double *xnuc, double eold, double s, double dedt, double time,
		   double *dsdt)
{
  double *n, ni, nb, ne;	// number densities [ particles per volume ]
  double ye;			// electron fraction ( electrons per baryon )
  double _temp, _tempold;	// temperatures [ K ]
  double _es, _is, _lambda;	// entropy per mass for electrons + positrons + radiation, protons
  double _ee, _deedt;		// energy per mass for electrons + positrons + radiation 
  double _ie, _diedt;		// energy per mass for ions
  double _s, _e, _ep;		// total entropy per mass, energy per mass, pressure
  double e;			// new energy per mass
  int iter, i;

  if(time == 0.0 || dedt == 0.0)
    {
      *dsdt = 0.0;
      return;
    }

  time *= 1e-3;

  if(rho < eos_table.rhoMin || rho > eos_table.rhoMax)
    {
      printf("Density out of table, rho: %g, table: [%g,%g]\n", rho, eos_table.rhoMin, eos_table.rhoMax);
      endrun(1337);
      return;
    }

  e = eold + dedt * time;

  n = (double *) malloc(EOS_NSPECIES * sizeof(double));
  ni = 0.0;
  ne = 0.0;
  for(i = 0; i < EOS_NSPECIES; i++)
    {
      n[i] = xnuc[i] * rho * AVOGADRO / eos_table.nuclearmasses[i];
      ni += n[i];
      ne += n[i] * eos_table.nuclearcharges[i];
    }
  nb = rho * AVOGADRO;
  ye = ne / nb;

  if(ye < eos_table.yeMin || ye > eos_table.yeMax)
    {
      printf("Electron fraction out of table, ye: %g, table: [%g,%g]\n", ye, eos_table.yeMin,
	     eos_table.yeMax);
      endrun(1337);
      return;
    }

  _diedt = 1.5 * ni * BOLTZMANN / rho;

  _temp = temp;
  _tempold = 0.0;
  iter = 0;
  while(iter < EOS_MAXITER)
    {
      if(rho <= eos_table.rhoMin)
	{
	  _ee = 0.0;
	  _deedt = 0.0;
	}
      else
	{
	  eos_trilinear_e(_temp, rho, ye, &_ee, &_deedt);
	}

      _ie = _diedt * _temp;
      _e = _ee + _ie;
      if(fabs(_e - e) <= (EOS_EPS * e))
	{
	  break;
	}

      _tempold = _temp;
      _temp = _temp - (_e - e) / (_deedt + _diedt);
      if(_temp <= eos_table.tempMin && _tempold == eos_table.tempMin)
	{
	  _temp = eos_table.tempMin;
	  break;
	}

      if(_temp >= eos_table.tempMax && _tempold >= eos_table.tempMax)
	{
	  _temp = eos_table.tempMax;
	  break;
	}

      _temp = _temp < eos_table.tempMax ? _temp : eos_table.tempMax;
      _temp = _temp > eos_table.tempMin ? _temp : eos_table.tempMin;
      iter++;
    }

  _is = 0.0;
  for(i = 0; i < EOS_NSPECIES; i++)
    {
      _lambda = PLANCK / sqrt(2.0 * EOS_PI * eos_table.nuclearmasses[i] / AVOGADRO * BOLTZMANN * _temp);
      _is += n[i] * (2.5 - log(n[i] * pow(_lambda, 3.0)));
    }
  _is *= BOLTZMANN / rho;


  if(rho <= eos_table.rhoMin)
    {
      _es = 0.0;
    }
  else
    {
      eos_trilinear(_temp, rho, ye, &_es, &_ee, &_ep);
    }

  _s = _is + _es;
  *dsdt = (_s - s) / time;

  free(n);
}

void eos_calc_egiven(double rho, double *xnuc, double e, double *temp, double *s)
{
  double *n, ni, nb, ne;	// number densities [ particles per volume ]
  double ye;			// electron fraction ( electrons per baryon )
  double _temp, _tempold;	// temperatures [ K ]
  double _es, _is, _lambda;	// entropy per mass for electrons + positrons + radiation, protons
  double _ee, _deedt;		// energy per mass for electrons + positrons + radiation 
  double _ie, _diedt;		// energy per mass for ions
  double _e, _ep;		// total entropy per mass, energy per mass, pressure
  int iter, i;

  if(rho < eos_table.rhoMin || rho > eos_table.rhoMax)
    {
      printf("Density out of table, rho: %g, table: [%g,%g]\n", rho, eos_table.rhoMin, eos_table.rhoMax);
      endrun(1337);
      return;
    }

  n = (double *) malloc(EOS_NSPECIES * sizeof(double));
  ni = 0.0;
  ne = 0.0;
  for(i = 0; i < EOS_NSPECIES; i++)
    {
      n[i] = xnuc[i] * rho * AVOGADRO / eos_table.nuclearmasses[i];
      ni += n[i];
      ne += n[i] * eos_table.nuclearcharges[i];
    }
  nb = rho * AVOGADRO;
  ye = ne / nb;

  if(ye < eos_table.yeMin || ye > eos_table.yeMax)
    {
      printf("Electron fraction out of table, ye: %g, table: [%g,%g]\n", ye, eos_table.yeMin,
	     eos_table.yeMax);
      endrun(1337);
      return;
    }

  _diedt = 1.5 * ni * BOLTZMANN / rho;

  // lets make a guess for the temperatures, assuming an ideal gas
  _temp = 2.0 / 3.0 * e * rho / (ni + ne) / BOLTZMANN;

  _tempold = 0.0;
  iter = 0;
  // maxiter is twice as large, as this function is only called once for the initial conditions
  while(iter < EOS_MAXITER * 2)
    {
      if(rho <= eos_table.rhoMin)
	{
	  _ee = 0.0;
	  _deedt = 0.0;
	}
      else
	{
	  eos_trilinear_e(_temp, rho, ye, &_ee, &_deedt);
	}

      _ie = _diedt * _temp;
      _e = _ee + _ie;
      if(fabs(_e - e) <= (EOS_EPS * e))
	{
	  break;
	}

      _tempold = _temp;
      _temp = _temp - (_e - e) / (_deedt + _diedt);
      if(_temp <= eos_table.tempMin && _tempold == eos_table.tempMin)
	{
	  _temp = eos_table.tempMin;
	  break;
	}

      if(_temp >= eos_table.tempMax && _tempold >= eos_table.tempMax)
	{
	  _temp = eos_table.tempMax;
	  break;
	}

      _temp = _temp < eos_table.tempMax ? _temp : eos_table.tempMax;
      _temp = _temp > eos_table.tempMin ? _temp : eos_table.tempMin;
      iter++;
    }

  _is = 0.0;
  for(i = 0; i < EOS_NSPECIES; i++)
    {
      _lambda = PLANCK / sqrt(2.0 * EOS_PI * eos_table.nuclearmasses[i] / AVOGADRO * BOLTZMANN * _temp);
      _is += n[i] * (2.5 - log(n[i] * pow(_lambda, 3.0)));
    }
  _is *= BOLTZMANN / rho;

  if(rho <= eos_table.rhoMin)
    {
      _es = 0.0;
    }
  else
    {
      eos_trilinear(_temp, rho, ye, &_es, &_ee, &_ep);
    }

  *temp = _temp;
  *s = _is + _es;

  free(n);
}

void eos_calc_sgiven(double rho, double *xnuc, double s, double *temp, double *p, double *e)
{
  double *n, ni, nb, ne;	// number densities [ particles per volume ]
  double ye;			// electron fraction ( electrons per baryon )
  double _temp, _tempold;	// temperatures [ K ]
  double _es, _desdt;		// entropy per mass for electrons + positrons + radiation 
  double _is, _disdt, _lambda;	// entropy per mass for ions
  double _s, _ee, _ep;		// total entropy per mass, energy per mass, pressure
  int iter, i;

  if(*temp == 0)
    return;

  if(rho < eos_table.rhoMin || rho > eos_table.rhoMax)
    {
      printf("Density out of table, rho: %g, table: [%g,%g]\n", rho, eos_table.rhoMin, eos_table.rhoMax);
      endrun(1337);
      return;
    }

  n = (double *) malloc(EOS_NSPECIES * sizeof(double));
  ni = 0.0;
  ne = 0.0;
  for(i = 0; i < EOS_NSPECIES; i++)
    {
      n[i] = xnuc[i] * rho * AVOGADRO / eos_table.nuclearmasses[i];
      ni += n[i];
      ne += n[i] * eos_table.nuclearcharges[i];
    }
  nb = rho * AVOGADRO;
  ye = ne / nb;

  if(ye < eos_table.yeMin || ye > eos_table.yeMax)
    {
      printf("Electron fraction out of table, ye: %g, table: [%g,%g]\n", ye, eos_table.yeMin,
	     eos_table.yeMax);
      endrun(1337);
      return;
    }

  _temp = *temp;
  _tempold = 0.0;
  iter = 0;
  while(iter < EOS_MAXITER)
    {
      if(rho <= eos_table.rhoMin)
	{
	  _es = 0.0;
	  _desdt = 0.0;
	}
      else
	{
	  eos_trilinear_s(_temp, rho, ye, &_es, &_desdt);
	}

      _is = 0.0;
      _disdt = 0.0;
      for(i = 0; i < EOS_NSPECIES; i++)
	{
	  _lambda = PLANCK / sqrt(2.0 * EOS_PI * eos_table.nuclearmasses[i] / AVOGADRO * BOLTZMANN * _temp);
	  _is += n[i] * (2.5 - log(n[i] * pow(_lambda, 3.0)));
	  _disdt += n[i];
	}
      _is *= BOLTZMANN / rho;
      _disdt *= 1.5 * BOLTZMANN / rho / _temp;

      _s = _is + _es;
      if(fabs(_s - s) <= (EOS_EPS * s))
	{
	  *temp = _temp;
	  break;
	}

      _tempold = _temp;
      _temp = _temp - (_s - s) / (_desdt + _disdt);
      if(_temp <= eos_table.tempMin && _tempold == eos_table.tempMin)
	{
	  *temp = eos_table.tempMin;
	  break;
	}

      if(_temp >= eos_table.tempMax && _tempold >= eos_table.tempMax)
	{
	  *temp = eos_table.tempMax;
	  break;
	}

      _temp = _temp < eos_table.tempMax ? _temp : eos_table.tempMax;
      _temp = _temp > eos_table.tempMin ? _temp : eos_table.tempMin;

      iter++;
    }

  if(rho <= eos_table.rhoMin)
    {
      *p = ni * BOLTZMANN * (*temp);
      *e = 1.5 * (*p) / rho;
    }
  else
    {
      eos_trilinear(*temp, rho, ye, &_s, &_ee, &_ep);
      *p = ni * BOLTZMANN * (*temp) + _ep;
      *e = 1.5 * ni * BOLTZMANN * (*temp) / rho + _ee;
    }

  free(n);
}

void eos_trilinear_s(double temp, double rho, double ye, double *s, double *dsdt)
{
  double logtemp, logrho;
  int itemp, irho, iye;
  unsigned int entry111, entry211, entry121, entry221, entry112, entry212, entry122, entry222;
  double w111, w211, w121, w221, w112, w212, w122, w222;
  double dx1, dx2, dy1, dy2, dz1, dz2;

  logtemp = log10(temp);
  logrho = log10(rho);

  itemp = (logtemp - eos_table.ltempMin) / eos_table.ltempDelta;
  irho = (logrho - eos_table.lrhoMin) / eos_table.lrhoDelta;
  iye = (ye - eos_table.yeMin) / eos_table.yeDelta;

  dx1 = (logtemp - eos_table.ltemp[itemp]) * eos_table.ltempDeltaI;
  dy1 = (logrho - eos_table.lrho[irho]) * eos_table.lrhoDeltaI;
  dz1 = (ye - eos_table.ye[iye]) * eos_table.yeDeltaI;
  dx2 = 1. - dx1;
  dy2 = 1. - dy1;
  dz2 = 1. - dz1;

  w111 = dx2 * dy2 * dz2;
  w112 = dx2 * dy2 * dz1;
  w121 = dx2 * dy1 * dz2;
  w122 = dx2 * dy1 * dz1;
  w211 = dx1 * dy2 * dz2;
  w212 = dx1 * dy2 * dz1;
  w221 = dx1 * dy1 * dz2;
  w222 = dx1 * dy1 * dz1;

  entry111 = (iye * eos_table.nrho + irho) * eos_table.ntemp + itemp;
  entry112 = ((iye + 1) * eos_table.nrho + irho) * eos_table.ntemp + itemp;
  entry121 = (iye * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp;
  entry122 = ((iye + 1) * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp;
  entry211 = (iye * eos_table.nrho + irho) * eos_table.ntemp + itemp + 1;
  entry212 = ((iye + 1) * eos_table.nrho + irho) * eos_table.ntemp + itemp + 1;
  entry221 = (iye * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp + 1;
  entry222 = ((iye + 1) * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp + 1;

  *s = eos_table.s[entry111] * w111 + eos_table.s[entry112] * w112 +
    eos_table.s[entry121] * w121 + eos_table.s[entry122] * w122 +
    eos_table.s[entry211] * w211 + eos_table.s[entry212] * w212 +
    eos_table.s[entry221] * w221 + eos_table.s[entry222] * w222;

  *dsdt = eos_table.dsdt[entry111] * w111 + eos_table.dsdt[entry112] * w112 +
    eos_table.dsdt[entry121] * w121 + eos_table.dsdt[entry122] * w122 +
    eos_table.dsdt[entry211] * w211 + eos_table.dsdt[entry212] * w212 +
    eos_table.dsdt[entry221] * w221 + eos_table.dsdt[entry222] * w222;
}

void eos_trilinear(double temp, double rho, double ye, double *s, double *e, double *p)
{
  double logtemp, logrho;
  int itemp, irho, iye;
  unsigned int entry111, entry211, entry121, entry221, entry112, entry212, entry122, entry222;
  double w111, w211, w121, w221, w112, w212, w122, w222;
  double dx1, dx2, dy1, dy2, dz1, dz2;

  logtemp = log10(temp);
  logrho = log10(rho);

  itemp = (logtemp - eos_table.ltempMin) / eos_table.ltempDelta;
  irho = (logrho - eos_table.lrhoMin) / eos_table.lrhoDelta;
  iye = (ye - eos_table.yeMin) / eos_table.yeDelta;

  dx1 = (logtemp - eos_table.ltemp[itemp]) * eos_table.ltempDeltaI;
  dy1 = (logrho - eos_table.lrho[irho]) * eos_table.lrhoDeltaI;
  dz1 = (ye - eos_table.ye[iye]) * eos_table.yeDeltaI;
  dx2 = 1. - dx1;
  dy2 = 1. - dy1;
  dz2 = 1. - dz1;

  w111 = dx2 * dy2 * dz2;
  w112 = dx2 * dy2 * dz1;
  w121 = dx2 * dy1 * dz2;
  w122 = dx2 * dy1 * dz1;
  w211 = dx1 * dy2 * dz2;
  w212 = dx1 * dy2 * dz1;
  w221 = dx1 * dy1 * dz2;
  w222 = dx1 * dy1 * dz1;

  entry111 = (iye * eos_table.nrho + irho) * eos_table.ntemp + itemp;
  entry112 = ((iye + 1) * eos_table.nrho + irho) * eos_table.ntemp + itemp;
  entry121 = (iye * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp;
  entry122 = ((iye + 1) * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp;
  entry211 = (iye * eos_table.nrho + irho) * eos_table.ntemp + itemp + 1;
  entry212 = ((iye + 1) * eos_table.nrho + irho) * eos_table.ntemp + itemp + 1;
  entry221 = (iye * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp + 1;
  entry222 = ((iye + 1) * eos_table.nrho + irho + 1) * eos_table.ntemp + itemp + 1;

  *s = eos_table.s[entry111] * w111 + eos_table.s[entry112] * w112 +
    eos_table.s[entry121] * w121 + eos_table.s[entry122] * w122 +
    eos_table.s[entry211] * w211 + eos_table.s[entry212] * w212 +
    eos_table.s[entry221] * w221 + eos_table.s[entry222] * w222;

  *e = eos_table.e[entry111] * w111 + eos_table.e[entry112] * w112 +
    eos_table.e[entry121] * w121 + eos_table.e[entry122] * w122 +
    eos_table.e[entry211] * w211 + eos_table.e[entry212] * w212 +
    eos_table.e[entry221] * w221 + eos_table.e[entry222] * w222;

  *p = eos_table.p[entry111] * w111 + eos_table.p[entry112] * w112 +
    eos_table.p[entry121] * w121 + eos_table.p[entry122] * w122 +
    eos_table.p[entry211] * w211 + eos_table.p[entry212] * w212 +
    eos_table.p[entry221] * w221 + eos_table.p[entry222] * w222;
}
#endif

void eos_checkswap(char *fname, int *swap)
{
  FILE *fd;
  size_t fsize, fpos;
  int blocksize, blockend;

  if(!(fd = fopen(fname, "r")))
    {
      printf("can't open file `%s' for reading eos table.\n", fname);
      endrun(123);
    }

  fseek(fd, 0, SEEK_END);
  fsize = ftell(fd);

  *swap = 0;
  fpos = 0;
  fseek(fd, 0, SEEK_SET);
  fread(&blocksize, sizeof(int), 1, fd);
  while(!feof(fd))
    {
      if(fpos + blocksize + 4 > fsize)
	{
	  *swap += 1;
	  break;
	}
      fpos += 4 + blocksize;
      fseek(fd, fpos, SEEK_SET);
      fread(&blockend, sizeof(int), 1, fd);
      if(blocksize != blockend)
	{
	  *swap += 1;
	  break;
	}
      fpos += 4;
      fread(&blocksize, sizeof(int), 1, fd);
    }

  if(*swap == 0)
    {
      fclose(fd);
      return;
    }

  fpos = 0;
  fseek(fd, 0, SEEK_SET);
  fread(&blocksize, sizeof(int), 1, fd);
  while(!feof(fd))
    {
      blocksize = eos_SwapInt(blocksize);
      if(fpos + blocksize + 4 > fsize)
	{
	  *swap += 1;
	  break;
	}
      fpos += 4 + blocksize;
      fseek(fd, fpos, SEEK_SET);
      fread(&blockend, sizeof(int), 1, fd);
      blockend = eos_SwapInt(blockend);
      if(blocksize != blockend)
	{
	  *swap += 1;
	  break;
	}
      fpos += 4;
      fread(&blocksize, sizeof(int), 1, fd);
    }

  fclose(fd);
}

#endif
