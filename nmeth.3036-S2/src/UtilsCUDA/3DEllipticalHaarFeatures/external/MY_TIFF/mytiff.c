
#include "mytiff.h"
#include <stdio.h>

void* Read_TIFF_Image (const char* filename, long long int *dims, int *ndims, int *imType)
{
	printf("ERROR: function Read_TIFF_Image not finished yet\n");
	exit(3);
	/*
//static void *read_array(Tiff_Reader *(*reader)(void *, int, string), string routine, void *file_source, int layer, Layer_Bundle *bundle)

//return (read_array(image_reader,"Read_Image",file_name,layer,NULL)); }


  long long int  depth;
  Array     *map;
  void      *image;

  File_Source series;
  depth = image_depth(file_source,routine);
  if (depth == 0)
	  return (NULL);
  series.source = file_source;
  if ((image = read_tiff(routine,depth,&series,reader,&map,layer,bundle)) == NULL) //map is null for simpole TIFF images
	  return (NULL);


  
  return (image);
  */
}


/****************************************************************************************
 *                                                                                      *
 *  CENTRAL READ ROUTINE                                                                *
 *                                                                                      *
 ****************************************************************************************/

  // There are depth planes in the tiff file, and each successive tif IFD is returned by
  //   calling reader_handler with the next plane number.  If layer < 0 then get all layers,
  //   otherwise load only the specified layer.  If there is a color-map and the first
  //   channel is CHAN_MAPPED then allocate and build a map, returning it in *pmap.  The
  //   originating external routine has name 'routine'.  One calls read_handler(source,-1,?) to
  //   take care of any epilogue activity for the reading of successive IFDs.
/*
static void *read_tiff(string routine, Dimn_Type depth, void *source, Tiff_Reader *(*read_handler)(void *, int, string), Array **pmap, int layer, Layer_Bundle *R(M(bundle)))

{ Indx_Type   *area, Area;
  int         *invert, Invert;
  Array      **array, *AVector;
  void       **planes, *Planes[16];

  Array       *map;
  int          cidx, lidx, lnum, nlayers;
  Dimn_Type    width, height;
  Array_Kind   kind;
  Value_Type   type;
  int          scale;
  Tiff_Reader *tif;
  Tiff_IFD    *ifd;
  Tiff_Image  *img;
  Tiff_Type    targ;
  string       text;
  int          count;
  Dimn_Type    dims[4];

  *pmap = NULL;

  tif = read_handler(source,0,routine);
  if (tif == NULL)
    return (NULL);

  ifd = Read_Tiff_IFD(tif);
  if (ifd == NULL)
    { string es = Tiff_Error_String();
      if (es != NULL)
        { if (grab_message())
            sprintf(Image_Estring,"Error reading Tif IFD: '%s' (%s)",es,routine);
          Tiff_Error_Release();
        }
      return (NULL);
    }
  img = Get_Tiff_Image(ifd);
  if (img == NULL)
    { string es = Tiff_Error_String();
      if (es != NULL)
        { if (grab_message())
            sprintf(Image_Estring,"Error reading Tif Image: '%s' (%s)",es,routine);
          Tiff_Error_Release();
        }
      return (NULL);
    }

  nlayers = 0;
  for (cidx = 0; cidx < img->number_channels; cidx += kind_size[kind])
    { kind = determine_kind(img,cidx);
      if (layer == nlayers)
        break;
      nlayers += 1;
    }

  if (layer >= 0)
    { if (cidx >= img->number_channels)
        { if (grab_message())
            sprintf(Image_Estring,"Layer %d does not exit in tiff (%s)",layer,routine);
          return (NULL);
        }
      lidx = cidx;
      nlayers += 1;
      area     = &Area - layer;
      invert   = &Invert - layer;
      array    = &AVector - layer;
    }
  else
    { int i;
      if (bundle->num_layers >= nlayers)
        { for (i = nlayers; i < bundle->num_layers; i++)
            Kill_Array(bundle->layers[i]);
          for (i = nlayers-1; i >= 0; i--)
            Free_Array(bundle->layers[i]);
        }
      else
        { for (i = bundle->num_layers-1; i >= 0; i--)
            Free_Array(bundle->layers[i]);
          bundle->layers =
              (Array **) Guarded_Realloc(bundle->layers,
                               (sizeof(Array *)+sizeof(Size_Type)+sizeof(int))*((size_t) nlayers),
                               routine);
          if (bundle->layers == NULL)
            { if (grab_message())
                sprintf(Image_Estring,"Out of memory (%s)",routine);
              return (NULL);
            }
        }
      bundle->num_layers = nlayers;
      lidx   = 0;
      area   = (Size_Type *) (bundle->layers + nlayers);
      invert = (int *) (area + nlayers);
      array  = bundle->layers;
    }

  { int i;

    if (img->number_channels > 16)
      { planes = (void **)
                   Guarded_Realloc(NULL,sizeof(void *)*((size_t) img->number_channels),routine);
        if (planes == NULL)
          { if (grab_message())
              sprintf(Image_Estring,"Out of memory (%s)",routine);
            return (NULL);
          }
      }
    else
      planes = Planes;
    for (i = 0; i < img->number_channels; i++)
      planes[i] = NULL;
  }

  width  = img->width;
  height = img->height;

  dims[0] = width;
  dims[1] = height;
  dims[2] = depth;

  if (layer >= 0)
    { cidx = lidx; lnum = layer; }
  else
    lnum = cidx = 0;
  for ( ; lnum < nlayers; lnum++, cidx += kind_size[kind])
    { kind         = determine_kind(img,cidx);
      type         = determine_type(img,cidx);
      scale        = img->channels[cidx]->scale;
      invert[lnum] = (img->channels[cidx]->interpretation == CHAN_WHITE);
      area[lnum]   = (((Indx_Type) width) * height) * type_size[type];
      array[lnum]  = Make_Array(kind,type,2 + (depth != 1),dims);
      array[lnum]->scale = scale;
    }

  map = NULL;
  if (img->channels[0]->interpretation == CHAN_MAPPED && layer <= 0)
    { int dom = (1 << img->channels[0]->scale);
 
      dims[0] = dom;
      map = Make_Array(RGB_KIND,UINT16_TYPE,1,dims);

      memcpy(map->data,img->map,(size_t) (map->size*type_size[UINT16_TYPE]));
    }

  if ((text = (string) Get_Tiff_Tag(ifd,TIFF_JF_ANO_BLOCK,&targ,&count)) == NULL)
    text = Empty_String;
  if (layer > 0)
    Set_Array_Text(array[layer],text);
  else
    Set_Array_Text(array[0],text);

  { int       i;
    Dimn_Type d;

    d = 0;
    while (1)
      { Tiff_Channel **chan = img->channels;

        if (layer >= 0)
          { cidx = lidx; lnum = layer; }
        else
          lnum = cidx = 0;
        for ( ; lnum < nlayers; lnum++)
          { Indx_Type base;

            kind = array[lnum]->kind;
            if (kind == RGB_KIND)
              for (i = 0; i < 3; i++)
                { base = channel_order(chan[cidx+i]);
                  planes[cidx+i] = ((char *) array[lnum]->data) + area[lnum]*(d+depth*base);
                }
            else if (kind == RGBA_KIND)
              for (i = 0; i < 4; i++)
                { base = channel_order(chan[cidx+i]);
                  planes[cidx+i] = ((char *) array[lnum]->data) + area[lnum]*(d+depth*base);
                }
            else // kind == PLAIN_KIND
              planes[cidx] = ((char *) array[lnum]->data) + area[lnum]*d;
            cidx += kind_size[kind];
          }
        Load_Tiff_Image_Planes(img,planes);

        Free_Tiff_Image(img);
        Free_Tiff_IFD(ifd);

        d += 1;
        if (d >= depth) break;

        tif = read_handler(source,1,routine);
        if (tif == NULL)
          goto cleanup;

        while (1)
          { int *tag;

            ifd = Read_Tiff_IFD(tif);
            if (ifd == NULL)
              { string es = Tiff_Error_String();
                if (es != NULL)
                  { if (grab_message())
                      sprintf(Image_Estring,"Error reading Tif IFD: '%s' (%s)",es,routine);
                    Tiff_Error_Release();
                  }
                goto cleanup;
              }
            tag = (int *) Get_Tiff_Tag(ifd,TIFF_NEW_SUB_FILE_TYPE,&targ,&count);
            if (tag == NULL || (*tag & TIFF_VALUE_REDUCED_RESOLUTION) == 0)
              break;
            Free_Tiff_IFD(ifd);
          }
        img = Get_Tiff_Image(ifd);
        if (img == NULL)
          { string es = Tiff_Error_String();
            if (es != NULL)
              { if (grab_message())
                  sprintf(Image_Estring,"Error reading Tif IFD: '%s' (%s)",es,routine);
                Tiff_Error_Release();
              }
            Free_Tiff_IFD(ifd);
            goto cleanup;
          }

        if (img->width != width || img->height != height)
          { if (grab_message())
              sprintf(Image_Estring,
                      "Planes of a stack are not of the same dimensions (%s)!",routine);
            Free_Tiff_Image(img);
            Free_Tiff_IFD(ifd);
            goto cleanup;
          }

        if (layer >= 0)
          { cidx = lidx; lnum = layer; }
        else
          lnum = cidx = 0;
        for ( ; lnum < nlayers; lnum++)
          { kind = array[lnum]->kind;
            if (determine_type(img,cidx) != array[lnum]->type ||
                determine_kind(img,cidx) != kind ||
                img->channels[cidx]->scale != array[lnum]->scale)
              { if (grab_message())
                  sprintf(Image_Estring,
                          "Planes of a stack are not of the same type (%s)!",routine);
                Free_Tiff_Image(img);
                Free_Tiff_IFD(ifd);
                goto cleanup;
              }
            cidx += kind_size[kind];
          }
      }
  }

  read_handler(source,-1,routine);

  if (layer >= 0)
    { cidx = lidx; lnum = layer; }
  else
    lnum = cidx = 0;
  for ( ; lnum < nlayers; lnum++)
    { if (invert[lnum])
        { double max;
          if (array[lnum]->type <= UINT32_TYPE)
            max = (double) ((((uint64) 1) << array[lnum]->scale) - 1);
          else if (array[lnum]->type <= INT32_TYPE)
            max = (double) ((((uint64) 1) << (array[lnum]->scale-1)) - 1);
          else
            max = 1.0;
          Scale_Array(array[lnum], -1., -max);
        }
      cidx += kind_size[array[lnum]->kind];
    }

  if (img->number_channels > 16)
    free(planes);

  *pmap = map;
  if (layer >= 0)
    return (array[layer]);
  else
    return (bundle);

cleanup:
  if (layer >= 0)
    { cidx = lidx; lnum = layer; }
  else
    lnum = cidx = 0;
  for ( ; lnum < nlayers; lnum++)
    Free_Array(array[lnum]);
  if (map != NULL) Free_Array(map);

  if (img->number_channels > 16)
    free(planes);

  *pmap = NULL;
  return (NULL);
}

*/