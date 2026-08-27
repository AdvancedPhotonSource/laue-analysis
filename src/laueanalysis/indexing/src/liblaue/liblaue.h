#ifndef LIBLAUE_H
#define LIBLAUE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define LAUE_API __attribute__((visibility("default")))
#else
#define LAUE_API
#endif

typedef struct laue_geometry laue_geometry;
typedef struct laue_crystal laue_crystal;

typedef enum {
    LAUE_OK = 0,
    LAUE_INVALID_ARGUMENT = 1,
    LAUE_OUT_OF_MEMORY = 2,
    LAUE_NUMERICAL_ERROR = 3,
    LAUE_INTERNAL_ERROR = 4
} laue_status;

typedef struct {
    char name[60];
    double x, y, z, occupancy;
} laue_atom;

typedef struct {
    int nx, ny;
    double size_x, size_y;
    double translation[3];
    double rotation_vector[3];
    double rotation[3][3];
    char detector_id[256];
} laue_detector_info;

typedef struct {
    int boxsize;
    double max_rfactor;
    int min_size;
    int min_separation;
    double threshold;
    double threshold_ratio;
    int peak_shape;
    int max_peaks;
    int smooth;
    const unsigned char *mask;
    int detect_binning;
} laue_peak_params;

typedef struct {
    double kev_max_calc;
    double kev_max_test;
    double angle_tolerance_deg;
    double cone_deg;
    int hkl_prefer[3];
    int max_data;
} laue_index_params;

typedef struct {
    double fit_x;
    double fit_y;
    double intens;
    double integral;
    double hwhm_x;
    double hwhm_y;
    double tilt;
    double chisq;
    double background;
    double qhat[3];
} laue_peak;

typedef struct {
    double euler_deg[3];
    double rotation[3][3];
    double recip[3][3];
    double goodness;
    double rms_error_deg;
    int n_indexed;
    int *hkl;
    int *pk_index;
    double *err_deg;
    double *energy_kev;
    double *pred_intens;
} laue_pattern;

typedef struct {
    int nx;
    int ny;
    int startx;
    int starty;
    int groupx;
    int groupy;
    double depth;
    double threshold_used;
    double total_sum;
    double sum_above_threshold;
    long num_above_threshold;
    int n_peaks;
    laue_peak *peaks;
    int n_patterns;
    int n_indexed;
    laue_pattern *patterns;
    int status;
    char message[256];
} laue_frame_result;

LAUE_API laue_geometry *laue_geometry_from_file(const char *path, char *err, size_t errlen);
LAUE_API void laue_geometry_free(laue_geometry *geometry);
LAUE_API laue_crystal *laue_crystal_from_file(const char *path, char *err, size_t errlen);
LAUE_API laue_crystal *laue_crystal_create(const char *name, int space_group,
                                            double a, double b, double c,
                                            double alpha_deg, double beta_deg, double gamma_deg,
                                            const laue_atom *atoms, size_t n_atoms,
                                            char *err, size_t errlen);
LAUE_API void laue_crystal_free(laue_crystal *crystal);
LAUE_API int laue_geometry_detector_count(const laue_geometry *geometry);
LAUE_API int laue_geometry_find_detector(const laue_geometry *geometry, const char *detector_id);
LAUE_API int laue_geometry_detector_info(const laue_geometry *geometry, int detector_index,
                                         laue_detector_info *info, char *err, size_t errlen);
LAUE_API int laue_find_peaks(const unsigned short *pixels, int nx, int ny,
                             const laue_peak_params *params, laue_frame_result *result);
LAUE_API int laue_pixels_to_q(const laue_geometry *geometry, int detector_index, laue_frame_result *result);
LAUE_API int laue_index(const laue_crystal *crystal, const laue_index_params *params,
                        laue_frame_result *result);
LAUE_API void laue_frame_result_free(laue_frame_result *result);
LAUE_API const char *laue_version(void);

#ifdef __cplusplus
}
#endif

#endif
