all: basic_usage basic_usage_autotune

basic_usage: basic_usage.o
	${MPICXX} -o $@ $< ${BUILD_LIBS}

basic_usage.o: basic_usage.cu
	${NVCC} ${NVFLAGS}  ${BUILD_INCLUDES} -c -o $@ $<

basic_usage_autotune: basic_usage_autotune.o
	${MPICXX} -o $@ $< ${BUILD_LIBS}

basic_usage_autotune.o: basic_usage_autotune.cu
	${NVCC} ${NVFLAGS} ${BUILD_INCLUDES} -c -o $@ $<

clean:
	rm -f basic_usage basic_usage.o
	rm -f basic_usage_autotune basic_usage_autotune.o
