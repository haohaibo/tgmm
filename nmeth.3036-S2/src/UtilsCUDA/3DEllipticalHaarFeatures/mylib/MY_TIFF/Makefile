CC     = gcc
CFLAGS = -O4 -Wall -Wextra
MFLAGS = -m64 -fPIC -D_LARGEFILE64_SOURCE
% MFLAGS = -arch i386 -fPIC

apps = tiffconvert tifftagger tiffRGB tiffshow mrc2tiff

modules = tiff.io.o tiff.image.o utilities.o

all: mytiff.o $(apps)

generator: generator.c
	$(CC) $(CFLAGS) -o generator generator.c

%.o: %.p generator
	./generator <$< >$*.c
	$(CC) -c $(CFLAGS) $(MFLAGS) $*.c

mytiff.o: tiff.io.o tiff.image.o
	ld -r -o mytiff.o tiff.io.o tiff.image.o

tiffconvert: app.convert.c utilities.o mytiff.o
	$(CC) $(CFLAGS) $(MFLAGS) -o tiffconvert app.convert.c utilities.o mytiff.o

tifftagger: app.tagger.c utilities.o mytiff.o
	$(CC) $(CFLAGS) $(MFLAGS) -o tifftagger app.tagger.c utilities.o mytiff.o

tiffRGB: app.color.c utilities.o mytiff.o
	$(CC) $(CFLAGS) $(MFLAGS) -o tiffRGB app.color.c utilities.o mytiff.o

tiffshow: app.show.c utilities.o mytiff.o
	$(CC) $(CFLAGS) $(MFLAGS) -o tiffshow app.show.c utilities.o mytiff.o

mrc2tiff: app.mrc2.c utilities.o mytiff.o
	$(CC) $(CFLAGS) $(MFLAGS) -o mrc2tiff app.mrc2.c utilities.o mytiff.o

clean:
	rm -f mytiff.o $(modules) $(modules:.o=.c) $(apps) mytiff.tar.gz generator

package:
	tar -zcf mytiff.tar.gz README INDEX *.h *.p app.*.c manager.awk Makefile
