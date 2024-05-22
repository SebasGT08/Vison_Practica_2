# Variables
CXX = g++
CXXFLAGS = -Wall `pkg-config --cflags opencv4 jsoncpp` -I/usr/include/jsoncpp
LDFLAGS = `pkg-config --libs opencv4 jsoncpp` -lcurl
TARGET = parte1

# Regla por defecto
all: $(TARGET)

# Compilar el archivo objetivo
$(TARGET): $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TARGET).cpp $(LDFLAGS)

# Limpiar archivos objeto y ejecutable
clean:
	rm -f $(TARGET)
