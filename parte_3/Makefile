# Variables
CXX = g++
CXXFLAGS = -Wall `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
TARGET = parte3

# Regla por defecto
all: $(TARGET)

# Compilar el archivo objetivo
$(TARGET): $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TARGET).cpp $(LDFLAGS)

# Limpiar archivos objeto y ejecutable
clean:
	rm -f $(TARGET)
