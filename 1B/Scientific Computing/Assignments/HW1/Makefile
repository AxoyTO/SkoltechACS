SRC = disubstr.cpp
BIN = disubstr.out
C++FLAGS = -o $(BIN) -std=c++17 -O3 -Wall
C++ =  g++
DATE = $(shell date +%d/%m/%y)
#COMPUTER ?= Desktop

.PHONY: all compile run clean

all: clean compile

compile:
	@echo " *  Compiling"
	-$(C++) $(SRC) $(C++FLAGS)
	@echo " *  Created on: $(DATE)"
	@echo " *  Author: Aksoy Tevfik Oguzhan"

run:
	./$(BIN)

clean:
	rm -rf $(BIN)