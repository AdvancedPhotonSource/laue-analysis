/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#ifndef LIBLAUE_INTERNAL_H
#define LIBLAUE_INTERNAL_H

#include <stdio.h>

#include "liblaue.h"
#include "readGeoN.h"

struct laue_geometry {
    struct geoStructure value;
    int has_wire;
};

#endif
