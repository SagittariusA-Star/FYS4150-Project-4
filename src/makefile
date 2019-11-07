UNAME := $(shell uname)


ifeq ($(UNAME), Linux) # Linux
$(info "Linux detected")
CPPflags= g++ -std=c++11 -O3
endif
ifeq ($(UNAME), Darwin) # Mac
$(info "MacOS detected")
CPPflags= mpic++ -std=c++11 -O3 -Wall
endif



LIB = -larmadillo


TESTPROG = testcode.out
DATA_A = data_a.out
DATA_B = data_b.out
DATA_C_D = data_c_d.out
DATA_E = data_e.out

all:	${TESTPROG} ${DATA_A} ${DATA_B} ${DATA_C_D} ${DATA_E}

${TESTPROG} :	   		tests-main.o  test-functions.o
					${CPPflags} tests-main.o test-functions.o -o ${TESTPROG}

tests-main.o :			tests-main.cpp
		        	${CPPflags} -c tests-main.cpp

test-functions.o :			test-functions.cpp
		        	${CPPflags} -c test-functions.cpp
				
${DATA_A}:        task_a.cpp
				${CPPflags} task_a.cpp ${LIB} -o ${DATA_A}

${DATA_B}:        task_b.cpp
				${CPPflags} task_b.cpp ${LIB} -o ${DATA_B}

${DATA_C_D}:        task_c_and_d.cpp
				${CPPflags} task_c_and_d.cpp ${LIB} -o ${DATA_C_D}

${DATA_E}:        task_e.cpp
				${CPPflags} task_e.cpp ${LIB} -o ${DATA_E}

clean:
		rm *.o

rmexecs:
		rm *.out