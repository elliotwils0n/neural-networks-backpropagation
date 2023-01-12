MAIN = src/main.cpp
TEST = src/test.cpp
SOURCES = src/Neuron.cpp src/Layer.cpp src/Network.cpp

ifdef OS # Windows specific env
	EXTENSION = exe
	LIBRARIES = -lsfml-system-2 -lsfml-window-2 -lsfml-graphics-2
	RM_COMMAND = del
	RUN_COMMAND = ./bp.exe
else
	EXTENSION = out
	LIBRARIES = -lsfml-system -lsfml-window -lsfml-graphics
	RM_COMMAND = rm
	RUN_COMMAND = export LD_LIBRARY_PATH=. && ./bp.out
endif

default: help

help:
	@echo 'Available commands:'
	@echo '  make test             - runs console test for random 3x3 image data'
	@echo '  make compile          - compiles the program'
	@echo '  make run [input=PATH] - runs the program (with optional input argument of image path)'
	@echo '  make clean            - removes compiled programs'

test:
	g++ -std=c++11 $(TEST) $(SOURCES) -o test.$(EXTENSION)
	./test.$(EXTENSION)
	$(RM_COMMAND) test.$(EXTENSION)

compile:
	g++ -std=c++11 $(MAIN) $(SOURCES) -o bp.$(EXTENSION) -I include -L . $(LIBRARIES)

run:
	$(RUN_COMMAND) $(input)

clean:
	$(RM_COMMAND) bp.$(EXTENSION)

# depricated:
LIBRARIES_WINDOWS = -lsfml-system-2 -lsfml-window-2 -lsfml-graphics-2
LIBRARIES_LINUX = -lsfml-system -lsfml-window -lsfml-graphics

compile-windows:
	g++ -std=c++11 $(MAIN) $(SOURCES) -o bp.exe -I include -L . $(LIBRARIES_WINDOWS)

run-windows:
	./bp.exe $(input)

compile-linux:
	g++ -std=c++11 $(MAIN) $(SOURCES) -o bp.out -I include -L . $(LIBRARIES_LINUX)

run-linux:
	export LD_LIBRARY_PATH=. && ./bp.out $(input)

