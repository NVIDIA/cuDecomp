all: basic_usage basic_usage_autotune

basic_usage: basic_usage.f90
	${MPIF90} ${BUILD_INCLUDES} -cuda -acc -cpp -Minfo=accel -o $@ $< ${BUILD_FLIBS}

basic_usage_autotune: basic_usage_autotune.f90
	${MPIF90} ${BUILD_INCLUDES} -cuda -acc -cpp -Minfo=accel -o $@ $< ${BUILD_FLIBS}

clean:
	rm -f basic_usage
	rm -f basic_usage_autotune
