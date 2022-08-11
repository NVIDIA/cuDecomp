tg: tg.o
	${MPICXX} -o $@ $< ${BUILD_LIBS}

tg.o: tg.cu
	${NVCC} ${NVFLAGS} ${BUILD_INCLUDES} -c -o $@ $<

clean:
	rm -f tg tg.o
