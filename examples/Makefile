EXAMPLETARGETS = cc
ifeq (${strip ${BUILD_FORTRAN}},1)
EXAMPLETARGETS += f90
endif

all: ${EXAMPLETARGETS}

.PHONY:cc
cc:
	cd cc/basic_usage; make
	cd cc/taylor_green; make

f90:
	cd fortran/basic_usage; make
	cd fortran/poisson; make
clean:
	cd cc/basic_usage; make clean
	cd cc/taylor_green; make clean
	cd fortran/basic_usage; make clean
	cd fortran/poisson; make clean
