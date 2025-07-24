#ifndef WireScanHeader
#define WireScanHeader

#include "WireScanDataTypesN.h"
#include "microHDF5.h"
#include "hardwareSpecific.h"
#include "readGeoN.h"

#ifndef MAX
#define MAX(X,Y) ( ((X)<(Y)) ? (Y) : (X) )
#endif
#ifndef MIN
#define MIN(X,Y) ( ((X)>(Y)) ? (Y) : (X) )
#endif


typedef short unsigned int BOOLEAN;

// struct cudaPara and cuda variables start from here
typedef struct
{
   double CKIX;
   double CKIY;
   double CKIZ;
   double UPDEPTHS;
   double UPDEPTHR;
   double IMDEPTHSIXE;
   double CWIREDIAMETER;
}cudaConstPara;

extern cudaConstPara paraPassed;

// cuda parameters end here

typedef struct
{
   float CKIX;
   float CKIY;
   float CKIZ;
   float UPDEPTHS;
   float UPDEPTHR;
   float IMDEPTHSIXE;
   float CWIREDIAMETER;
}floatcudaConstPara;

extern floatcudaConstPara floatparaPassed;

/* data structures containing information for the wire scan */
extern ws_calibration calibration;
extern ws_imaging_parameters imaging_parameters;
extern ws_image_set image_set;
extern ws_user_preferences user_preferences;

extern gsl_matrix * intensity_map;

/* struct HDF5_Header first_header; */
extern struct HDF5_Header in_header;
extern struct HDF5_Header output_header;
extern struct geoStructure geoIn;

extern int		verbose;							/* default to 0 */
extern float	percent;							/* default to 100 */
extern int		cutoff;								/* default to 0 */
extern int		AVAILABLE_RAM_MiB;					/* default to 128 */
extern int		detNum;								/* detector number, default to 0 */
extern char	distortionPath[FILENAME_MAX];		/* full path to the distortion map */
extern char	depthCorrectStr[FILENAME_MAX];		/* full path to the depth correction map */

#endif
