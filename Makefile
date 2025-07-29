CXX = clang++
CXXFLAGS = -std=c++17 -g -Iinclude

SRC = src/main.cpp src/matrix.cpp src/layer_dense.cpp src/activation.cpp src/loss.cpp
OUT = build/neuralnet

all: $(OUT)

$(OUT): $(SRC)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT)

run: all
	./$(OUT)

clean:
	rm -rf build




