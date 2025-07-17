#ifndef WireScanHeader
#define WireScanHeader

#include "WireScanDataTypesN.h"
#include "microHDF5.h"
#include "hardwareSpecific.h"


#ifndef MAX
#define MAX(X,Y) ( ((X)<(Y)) ? (Y) : (X) )
#endif
#ifndef MIN
#define MIN(X,Y) ( ((X)>(Y)) ? (Y) : (X) )
#endif


typedef short unsigned int BOOLEAN;

/* data structures containing information for the wire scan */
ws_calibration calibration;
ws_imaging_parameters imaging_parameters;
ws_image_set image_set;
ws_user_preferences user_preferences;

gsl_matrix * intensity_map;
gsl_matrix * intensity_norm;

/* extra data-holders defined for parallel version */
vvector p_read_buffer[2]; 	// buffer to hold input data for parallel execution
vvector p_write_buffer[2]; 	// buffer to hold output data for parallel execution
int p_ibuff;				// index to which buffer to be worked on
int rows_default;			// limit on # of rows per stripe, default 64
int	NUM_THREADS;			// default to 1/

struct HDF5_Header in_header;
//struct HDF5_Header first_header;
struct HDF5_Header output_header;
struct geoStructure geoIn;

int		verbose;							/* default to 0 */
float	percent;							/* default to 100 */
int		cutoff;								/* default to 0 */
int		AVAILABLE_RAM_MiB;					/* default to 128 */
int		cosmic;								/* default to 0 */
int		detNum;								/* detector number, default to 0 */
char	distortionPath[FILENAME_MAX];		/* full path to the distortion map */
float	norm_exponent;						/* exponent to use on normalization image */
float	norm_threshold;						/* threshold to use on normalization image */
float	norm_rescale;						/* if using norm_exponent, multiply result by this */

#endif
