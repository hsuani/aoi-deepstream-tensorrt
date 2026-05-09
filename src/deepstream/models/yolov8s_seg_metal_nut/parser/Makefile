DEEPSTREAM_DIR ?= /opt/nvidia/deepstream/deepstream
CUDA_VER ?= 12.8
CC := g++
CFLAGS := -Wall -O2 -std=c++14 -shared -fPIC
CFLAGS += -I$(DEEPSTREAM_DIR)/sources/includes -I/usr/local/cuda/include
LIBS := -lnvinfer
LFLAGS := -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES := nvdsinfer_custombboxparser_yolov8s_seg_metal_nut.cpp
TARGET_LIB := libnvdsinfer_yolov8s_seg_metal_nut_parser.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
