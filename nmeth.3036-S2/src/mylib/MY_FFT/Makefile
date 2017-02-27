CC     = gcc
CFLAGS = -O4 -Wall -Wextra
MFLAGS = -m64 -msse3 -fPIC
% MFLAGS = -arch i386 -msse3 -fPIC

modules = fft.F.o fft.D.o myfft.o

all: $(modules)

fft.F.o: fft.F.c fft.F.h
	$(CC) -c $(CFLAGS) $(MFLAGS) fft.F.c

fft.D.o: fft.D.c fft.D.h
	$(CC) -c $(CFLAGS) $(MFLAGS) fft.D.c

myfft.o: fft.F.c fft.F.h fft.D.c fft.D.h
	$(CC) -c $(CFLAGS) $(MFLAGS) myfft.c

test:
	gcc $(CFLAGS) $(MFLAGS) test.fft.c myfft.o
	a.out
	gcc $(CFLAGS) $(MFLAGS) test.norm.F.c myfft.o
	a.out
	gcc $(CFLAGS) $(MFLAGS) test.norm.D.c myfft.o
	a.out
	rm a.out

clean:
	rm -f myfft.o fft.F.o fft.D.o a.out myfft.tar.gz

package:
	tar -zcf myfft.tar.gz README *.h *.c Makefile
