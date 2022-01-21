# --- MACROS
# define program name
MAIN= subdivide

# define the C compiler to use
CC= nvcc

# define any compile-time flags
CFLAGS= -O2 -rdc=true -g -G -allow-unsupported-compiler


# define any libraries to link into executable
LIBS= 

# define C source files
SRCS= ${wildcard src/*.cu} ${wildcard src/**/*.cu} ${wildcard src/**/**/*.cu}

# define C header files
HDRS= ${wildcard src/*.cuh} ${wildcard src/**/*.cuh} ${wildcard src/**/**/*.cuh}

# --- TARGETS
all: ${MAIN}

#Builds the program
${MAIN}: ${SRCS} ${HDRS}
	@echo #
	@echo "-- BUILDING PROGRAM --"
	${CC} ${SRCS} ${CFLAGS} ${LIBS} -o ${MAIN}.exe

clean:
	@echo #
	@echo "-- CLEANING PROJECT FILES --"
	$(RM) *.o ${MAIN}