tg: tg.o
	${MPICXX} -o $@ $< ${LIBS}

tg.o: tg.cu
	${NVCC} ${NVFLAGS} ${DEFINES} ${EXTRA_DEFINES} ${INCLUDES} -c -o $@ $<

clean:
	rm -f tg tg.o
