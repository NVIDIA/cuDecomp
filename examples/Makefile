EXAMPLETARGETS = cc
ifeq (${strip ${BUILD_FORTRAN}},1)
EXAMPLETARGETS += f90
endif

all: ${EXAMPLETARGETS}

.PHONY:cc
cc:
	cd cc/basic_usage; make CONFIGFILE=${CONFIGFILE}
	cd cc/taylor_green; make CONFIGFILE=${CONFIGFILE}

f90:
	cd fortran/basic_usage; make CONFIGFILE=${CONFIGFILE}
	cd fortran/poisson; make CONFIGFILE=${CONFIGFILE}
clean:
	cd cc/basic_usage; make clean
	cd cc/taylor_green; make clean
	cd fortran/basic_usage; make clean
	cd fortran/poisson; make clean
