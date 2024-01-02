COMPILER:=gcc
OPTIONS:=-g -pedantic -w -Wall -Wextra -Wshadow -Wconversion -Wunreachable-code
COMPILE:=$(COMPILER) $(OPTIONS)

SRC:=src
HDR:=headers
BUILD:=build
MAIN:=$(SRC)/main.c

all: program $(HDR)/lib.h $(HDR)/structs.h

$(BUILD)/%.o: $(SRC)/%.c
	$(COMPILE) -c $< -o $@ 

program: $(BUILD)/cond.o $(BUILD)/layer.o $(BUILD)/mat_ops.o $(BUILD)/net.o $(BUILD)/ops.o $(BUILD)/read.o | $(MAIN) build
	$(COMPILE) $(MAIN) $^ -o $(BUILD)/$@

clean:
	rm -rf $(BUILD)

build: 
	mkdir -p $(BUILD)



.PHONY: build clean