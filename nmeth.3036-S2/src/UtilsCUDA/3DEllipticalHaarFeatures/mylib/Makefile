CC     = gcc
CFLAGS = -O4 -Wall -Wextra
MFLAGS = -m64 -fPIC -D_LARGEFILE64_SOURCE
% MFLAGS = -arch i386 -fPIC

modules = cdf.o fct.min.o filters.o linear.algebra.o array.o connectivity.o \
          region.o fct.root.o image.o utilities.o snake.o  hash.o level.set.o    \
          water.shed.o fft.o draw.o histogram.o swc.o paths.o mylib.o svg.o

all: mylib

generator: generator.c
	$(CC) $(CFLAGS) -o generator generator.c

%.o: %.p generator
	./generator <$< >$*.c
	$(CC) -c $(CFLAGS) $(MFLAGS) $*.c

mylib: $(modules)
	cd MY_TIFF   && $(MAKE) && cd ..
	cd MY_FFT    && $(MAKE) && cd ..
	libtool -static -o libmylib.a $(modules) MY_FFT/myfft.o MY_TIFF/mytiff.o

clean:
	cd MY_TIFF   && $(MAKE) clean && cd ..
	cd MY_FFT    && $(MAKE) clean && cd ..
	rm -f libmylib.a $(modules:.o=.c) $(modules) generator

package:
	cd MY_TIFF   && $(MAKE) clean && cd ..
	cd MY_FFT    && $(MAKE) clean && cd ..
	rm -f libmylib.a $(modules:.o=.c) $(modules)
	tar -czf mylib.tar.gz DOCUMENTS/PDFS/*.pdf DOCUMENTS/PPT/*.ppt DOCUMENTS/README \
                              DOCUMENTS/HTML/* DOCUMENTS/HTML/IMAGES/* DOCUMENTS/BUILDME \
                              MY_TIFF/* MY_FFT/* *.p *.h *.awk Makefile CMakeLists.txt
