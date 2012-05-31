#ifdef EOS_DEGENERATE

#define EOS_MAXITER 40

// basic constants in cgs units
#define EOS_PI 3.14159265359 // Pi
#define EOS_EPS 1.0e-13

void eos_init();
void eos_deinit();
void eos_calc_egiven2( double rho, double *xnuc, double e, double *temp, double *p );
void eos_trilinear_e( double temp, double rho, double ye, double *e, double *dedt );
void eos_trilinear2( double temp, double rho, double ye, double *e, double *p );
void eos_checkswap( char* fname, int *swap );

#ifndef EOS_ENERGY
void eos_calc_dsdt( double temp, double rho, double *xnuc, double eold, double s, double dedt, double time, double *dsdt );
void eos_calc_egiven( double rho, double *xnuc, double e, double *temp, double *s );
void eos_calc_sgiven( double rho, double *xnuc, double s, double *temp, double *p, double *e );
void eos_trilinear_s( double temp, double rho, double ye, double *s, double *dsdt );
void eos_trilinear( double temp, double rho, double ye, double *s, double *e, double *p );
#endif

struct eos_table {
  double *nuclearmasses;
  double *nuclearcharges;
  int ntemp, nrho, nye;
  double tempMin, tempMax, ltempMin, ltempMax, ltempDelta, ltempDeltaI;
  double rhoMin, rhoMax, lrhoMin, lrhoMax, lrhoDelta, lrhoDeltaI;
  double yeMin, yeMax, yeDelta, yeDeltaI;
  double *ltemp, *lrho, *ye; 
   
  double *p;           /* pressure */
  double *e;           /* energy per mass */
  double *dedt;        /* derivative energy with temperature */
#ifndef EOS_ENERGY
  double *s;           /* entropy per mass */
  double *dsdt;        /* derivative entropy with temperature */ 
#endif
} eos_table;

inline double eos_SwapDouble( double Val ) {
  double nVal;
  int i;
  const char* readFrom = ( const char* ) &Val;
  char * writeTo = ( ( char* ) &nVal ) + sizeof( nVal );
  for (i=0; i<sizeof( Val ); ++i) {
    *( --writeTo ) = *( readFrom++ );
  }
  return nVal;
}

inline int eos_SwapInt( int Val ) {
  int nVal;
  int i;
  const char* readFrom = ( const char* ) &Val;
  char * writeTo = ( ( char* ) &nVal ) + sizeof( nVal );
  for (i=0; i<sizeof( Val ); ++i) {
    *( --writeTo ) = *( readFrom++ );
  }
  return nVal;
}

inline double eos_calcYe( double *xnuc ) {
  double ye, atot, ztot;
  int i;

  atot = 0.0;
  ztot = 0.0;
  for (i=0; i<EOS_NSPECIES; i++) {
    atot += xnuc[i] * eos_table.nuclearmasses[i];
    ztot += xnuc[i] * eos_table.nuclearcharges[i];
  }
  ye = ztot / atot;  

  return ye;
}

#endif
