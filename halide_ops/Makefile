include ./Makefile.inc

OPS = $(BIN)/$(HL_TARGET)/kernel_weighting_cpu.a \
      $(BIN)/$(HL_TARGET)/kernel_weighting_grad_cpu.a \
	  $(BIN)/$(HL_TARGET)/scatter2gather_cpu.a \

# Check whether we have cuda installed, if add the CUDA ops as dependencies
ifeq ($(shell which nvcc),)
HAS_CUDA = 0
CUDA_OPS =
else
HAS_CUDA = 1
CUDA_OPS = $(BIN)/$(HL_TARGET)/kernel_weighting_cuda.a \
      	   $(BIN)/$(HL_TARGET)/kernel_weighting_grad_cuda.a \
		   $(BIN)/$(HL_TARGET)/scatter2gather_cuda.a
endif

EXT_LIB=$(BIN)/lib
PYTHON ?= python3
PIP ?= pip3

all: $(BIN)/.wrapper

CUDA_TARGET ?= host-cuda-cuda_capability_61-user_context

# Build the python PyTorch extension that links against the Halide operators
$(BIN)/.wrapper: $(OPS) $(CUDA_OPS) setup.py
	@mkdir -p $(EXT_LIB)
	@HAS_CUDA=$(HAS_CUDA) \
	      HALIDE_DISTRIB_PATH=$(HALIDE_DISTRIB_PATH) \
	      BIN=$(BIN)/$(HL_TARGET) \
	      PYTHONPATH=$(EXT_LIB):${PYTHONPATH} \
	      $(PIP) install . -t $(EXT_LIB)
	@touch $(BIN)/.wrapper

# Generate the CPU version of the op ------------------------------------------
$(BIN)/%/kernel_weighting_cpu.a: $(GENERATOR_BIN)/kernel_weighting.generator
	@mkdir -p $(@D)
	@echo Producing CPU operator
	@$^ -g kernel_weighting \
		-f kernel_weighting_cpu \
		-e static_library,c_header,pytorch_wrapper \
		-o $(@D) \
		target=$* \
		auto_schedule=false

$(BIN)/%/kernel_weighting_grad_cpu.a: $(GENERATOR_BIN)/kernel_weighting.generator
	@mkdir -p $(@D)
	@echo Producing CPU gradient
	@$^ -g kernel_weighting_grad \
		-f kernel_weighting_grad_cpu \
		-e static_library,c_header,pytorch_wrapper \
		-o $(@D) \
		target=$* \
		auto_schedule=false

$(BIN)/%/scatter2gather_cpu.a: $(GENERATOR_BIN)/scatter2gather.generator
	@mkdir -p $(@D)
	@echo Producing CPU operator
	@$^ -g scatter2gather \
		-f scatter2gather_cpu \
		-e static_library,c_header,pytorch_wrapper \
		-o $(@D) \
		target=$* \
		auto_schedule=false

# Generate the GPU version of the op ------------------------------------------
$(BIN)/%/kernel_weighting_cuda.a: $(GENERATOR_BIN)/kernel_weighting.generator
	@mkdir -p $(@D)
	@echo Producing CUDA operator
	@$^ -g kernel_weighting \
		-f kernel_weighting_cuda \
		-e static_library,c_header,pytorch_wrapper \
		-o $(@D) \
		target=$(CUDA_TARGET) \
		auto_schedule=false

$(BIN)/%/kernel_weighting_grad_cuda.a: $(GENERATOR_BIN)/kernel_weighting.generator
	@mkdir -p $(@D)
	@echo "Producing CUDA gradient"
	@$^ -g kernel_weighting_grad \
		-f kernel_weighting_grad_cuda \
		-e static_library,c_header,pytorch_wrapper \
		-o $(@D) \
		target=$(CUDA_TARGET) \
		auto_schedule=false

$(BIN)/%/scatter2gather_cuda.a: $(GENERATOR_BIN)/scatter2gather.generator
	@mkdir -p $(@D)
	@echo Producing CUDA operator
	@$^ -g scatter2gather \
		-f scatter2gather_cuda \
		-e static_library,c_header,pytorch_wrapper \
		-o $(@D) \
		target=$(CUDA_TARGET) \
		auto_schedule=false

# Build the Halide generator for the operator
$(GENERATOR_BIN)/kernel_weighting.generator: src/kernel_weighting.cpp $(GENERATOR_DEPS)
	@echo Building kernel_weighting Generator
	@mkdir -p $(@D)
	@$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) \
		-o $@ $(LIBHALIDE_LDFLAGS) $(HALIDE_SYSTEM_LIBS)

$(GENERATOR_BIN)/scatter2gather.generator: src/scatter2gather.cpp $(GENERATOR_DEPS)
	@echo Building scatter2gather Generator
	@mkdir -p $(@D)
	@$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) \
		-o $@ $(LIBHALIDE_LDFLAGS) $(HALIDE_SYSTEM_LIBS)

clean:
	rm -rf $(BIN) __pycache__
