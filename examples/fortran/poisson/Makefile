poisson: poisson.f90
	${MPIF90} -cuda -cpp ${BUILD_INCLUDES} -o $@ $< ${BUILD_FLIBS}

clean:
	rm -f poisson
