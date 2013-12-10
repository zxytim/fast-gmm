#
# $File: Makefile
# $Date: Tue Dec 10 11:57:58 2013 +0800
#
# A single output portable Makefile for
# simple c++ project

OBJ_DIR = obj
BIN_DIR = bin
TARGET = gmm

CXX = g++
#CXX = clang++

BIN_TARGET = $(BIN_DIR)/$(TARGET)
PROF_FILE = $(BIN_TARGET).prof

INCLUDE_DIR = -I src
#LDFLAGS = -L /home/zhanpeng/.local/lib -lGClasses
#LDFLAGS += -lprofiler
#DEFINES += -D__DEBUG
#DEFINES += -D__DEBUG_CHECK


# CXXFLAGS += -O3
CXXFLAGS += -g -O0
#CXXFLAGS += -pg
CXXFLAGS += #$(DEFINES)
CXXFLAGS += -std=c++11
#CXXFLAGS += -ansi
CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(INCLUDE_DIR)
CXXFLAGS += $(LDFLAGS)
#CXXFLAGS += $(shell pkg-config --libs --cflags opencv)
#CXXFLAGS += -pthread
CXXFLAGS += -lpthread
#CXXFLAGS += -fopenmp


#CC = /usr/share/clang/scan-build/ccc-analyzer
#CXX = /usr/share/clang/scan-build/c++-analyzer
CXXSOURCES = $(shell find src/ -name "*.cc")
OBJS = $(addprefix $(OBJ_DIR)/,$(CXXSOURCES:.cc=.o))
DEPFILES = $(OBJS:.o=.d)

.PHONY: all clean run rebuild gdb

all: $(BIN_TARGET)

$(OBJ_DIR)/%.o: %.cc
	@echo "[cc] $< ..."
	@$(CXX) -c $< $(CXXFLAGS) -o $@

$(OBJ_DIR)/%.d: %.cc
	@mkdir -pv $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(INCLUDE_DIR) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cc=.o) $(OBJ_DIR)/$(<:.cc=.d)" "$<" > "$@"

sinclude $(DEPFILES)

$(BIN_TARGET): $(OBJS)
	@echo "[link] $< ..."
	@mkdir -p $(BIN_DIR)
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(CXXFLAGS)
	@echo have a nice day!

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(BIN_TARGET)
	./$(BIN_TARGET)

rebuild:
	+@make clean
	+@make

gdb: $(BIN_TARGET)
	gdb ./$(BIN_TARGET)

run-prof $(PROF_FILE): $(BIN_TARGET)
	CPUPROFILE=$(PROF_FILE) CPUPROFILE_FREQUENCY=1000 ./$(BIN_TARGET)

show-prof: $(PROF_FILE)
	google-pprof --text $(BIN_TARGET) $(PROF_FILE) | tee prof.txt

