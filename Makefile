ITENSOR_DIR=/home/per/phd-studies/itensor
include $(ITENSOR_DIR)/this_dir.mk
include $(ITENSOR_DIR)/options.mk

################################################################
#Options --------------

HEADERS=paralleldo.h mllib/mnist.h util.h single.h

ifdef app
APP=$(app)
else
APP=single
APP=fulltest
APP=separate_fulltest
APP=linear
APP=fixedL
endif


#################################################################

OBJECTS=$(APP).o

#Mappings --------------
GOBJECTS=$(patsubst %,.debug_objs/%, $(OBJECTS))

#Define Flags ----------
# If running gprof
# CCFLAGS += -pg

#Rules ------------------

%.o: %.cc $(HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

%.o: %.cpp $(HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cpp $(HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

default: build

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) 
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o $(APP) $(LIBFLAGS)

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o $(APP)-g $(LIBGFLAGS)

mkdebugdir:
	mkdir -p .debug_objs

clean:
	rm -fr *.o $(APP) $(APP)-g .debug_objs fulltest single fixedL linear separate_fulltest
