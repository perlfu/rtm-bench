targets = rtm-bench
cc = gcc -O2 -Wall
link = -lpthread -lrt
# remove -lrt to compile on OS X

all: $(targets)

rtm-bench: rtm-bench.c rtm.h
	$(cc) rtm-bench.c -o rtm-bench $(link)

clean:
	rm -f rtm-bench

