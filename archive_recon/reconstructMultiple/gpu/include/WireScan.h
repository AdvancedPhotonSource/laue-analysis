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

cudaConstPara paraPassed;

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

floatcudaConstPara floatparaPassed;

/* data structures containing information for the wire scan */
ws_calibration calibration;
ws_imaging_parameters imaging_parameters;
ws_image_set image_set;
ws_user_preferences user_preferences;

gsl_matrix * intensity_map;

/* struct HDF5_Header first_header; */
struct HDF5_Header in_header;
struct HDF5_Header output_header;
struct geoStructure geoIn;

int		verbose;							/* default to 0 */
float	percent;							/* default to 100 */
int		cutoff;								/* default to 0 */
int		AVAILABLE_RAM_MiB;					/* default to 128 */
int		detNum;								/* detector number, default to 0 */
char	distortionPath[FILENAME_MAX];		/* full path to the distortion map */
char	depthCorrectStr[FILENAME_MAX];		/* full path to the depth correction map */

#endif
