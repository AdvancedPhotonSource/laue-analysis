/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef limit
	#define limit( a, b, c ) min(max((a),(b)),(c))
/*	#define limit( a, b, c ) ( (( ((a) > (b)) ? (a) : (b) ) < (c)) ? ( ((a) > (b)) ? (a) : (b) ) : (c) ) */
#endif

#ifndef mod
#define mod(a,b) (((a-=((int)(a/b))*b)<0)? (a+=b):(a)) 
#endif
