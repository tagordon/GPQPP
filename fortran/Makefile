all: envelopes.o
	gfortran -shared -Ofast envelopes.o -o wrapper.so

envelopes.o:
	gfortran -g -shared -Ofast -unroll=12 -c envelopes.f90

clean:
	rm *.o
	rm *.so
	rm *.mod
