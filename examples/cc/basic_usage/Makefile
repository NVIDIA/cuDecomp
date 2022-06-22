all: basic_usage basic_usage_autotune

basic_usage: basic_usage.o
	${MPICXX} -o $@ $< ${LIBS}

basic_usage.o: basic_usage.cu
	${NVCC} ${NVFLAGS} ${DEFINES} ${EXTRA_DEFINES} ${INCLUDES} -c -o $@ $<

basic_usage_autotune: basic_usage_autotune.o
	${MPICXX} -o $@ $< ${LIBS}

basic_usage_autotune.o: basic_usage_autotune.cu
	${NVCC} ${NVFLAGS} ${DEFINES} ${EXTRA_DEFINES} ${INCLUDES} -c -o $@ $<

clean:
	rm -f basic_usage basic_usage.o
	rm -f basic_usage_autotune basic_usage_autotune.o
