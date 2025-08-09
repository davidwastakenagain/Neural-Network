CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -Iinclude
TARGET = main
SRCDIR = src
INCDIR = include
BUILDDIR = build

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/main.o: $(SRCDIR)/main.cpp $(INCDIR)/matrix.h $(INCDIR)/layer_dense.h $(INCDIR)/activation.h $(INCDIR)/loss.h $(INCDIR)/mnist_loader.h

$(BUILDDIR)/matrix.o: $(SRCDIR)/matrix.cpp $(INCDIR)/matrix.h

$(BUILDDIR)/layer_dense.o: $(SRCDIR)/layer_dense.cpp $(INCDIR)/layer_dense.h $(INCDIR)/matrix.h

$(BUILDDIR)/activation.o: $(SRCDIR)/activation.cpp $(INCDIR)/activation.h $(INCDIR)/matrix.h

$(BUILDDIR)/loss.o: $(SRCDIR)/loss.cpp $(INCDIR)/loss.h $(INCDIR)/matrix.h

$(BUILDDIR)/mnist_loader.o: $(SRCDIR)/mnist_loader.cpp $(INCDIR)/mnist_loader.h $(INCDIR)/matrix.h

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -rf $(BUILDDIR)

.PHONY: all clean