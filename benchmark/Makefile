all: benchmark_r2c benchmark_c2c benchmark_r2c_f benchmark_c2c_f

benchmark_r2c: benchmark_r2c.o
	${MPICXX} -o $@ $< ${BUILD_LIBS}

benchmark_c2c: benchmark_c2c.o
	${MPICXX} -o $@ $< ${BUILD_LIBS}

benchmark_r2c_f: benchmark_r2c_f.o
	${MPICXX} -o $@ $< ${BUILD_LIBS}

benchmark_c2c_f: benchmark_c2c_f.o
	${MPICXX} -o $@ $< ${BUILD_LIBS}

benchmark_r2c.o: benchmark.cu
	${NVCC} ${DEFINES} ${NVFLAGS} ${BUILD_INCLUDES} -DR2C -c -o $@ $<

benchmark_c2c.o: benchmark.cu
	${NVCC} ${DEFINES} ${NVFLAGS} ${BUILD_INCLUDES} -DC2C -c -o $@ $<

benchmark_r2c_f.o: benchmark.cu
	${NVCC} ${DEFINES} ${NVFLAGS} ${BUILD_INCLUDES} -DR2C -DUSE_FLOAT -c -o $@ $<

benchmark_c2c_f.o: benchmark.cu
	${NVCC} ${DEFINES} ${NVFLAGS} ${BUILD_INCLUDES} -DC2C -DUSE_FLOAT -c -o $@ $<

clean:
	rm -f *.o benchmark_r2c benchmark_c2c benchmark_r2c_f benchmark_c2c_f
