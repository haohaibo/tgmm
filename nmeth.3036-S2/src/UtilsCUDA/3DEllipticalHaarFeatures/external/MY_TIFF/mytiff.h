/*
 * mytiff.h
 *
 *  Created on: December 18th, 2012
 *      Author: amatf
 *
 *
 *      @brief  Light-weight interface to read simple TIFF images using MY_TIFF
 * library by Gene Myers. The functions here are translated from Mylib by Gene
 * Myer's so we don't need to import the whole library if we only want to read
 * TIFF images.
 */

#ifndef MY_TIFF_LIGHTWEIGHT_H_
#define MY_TIFF_LIGHTWEIGHT_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  UINT8_IMTYPE = 0,
  UINT16_IMTYPE = 1,
  UINT32_IMTYPE = 2,
  UINT64_IMTYPE = 3,
  INT8_IMTYPE = 4,
  INT16_IMTYPE = 5,
  INT32_IMTYPE = 6,
  INT64_IMTYPE = 7,
  FLOAT32_IMTYPE = 8,
  FLOAT64_IMTYPE = 9
} Value_ImType;

/*
 @brief  Main routine to read file. Returns NULL if there was an issue reading
 file

 imType: follows my_lib convention

*/
void *Read_TIFF_Image(const char *filename, long long int *dims, int *ndims,
                      int *imType);

#ifdef __cplusplus
}
#endif

#endif