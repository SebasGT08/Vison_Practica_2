#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Función para agregar texto a una imagen
void agregarTexto(Mat &imagen, const string &texto, const Point &posicion) {
    putText(imagen, texto, posicion, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
}

// Función para aplicar operaciones morfológicas a una imagen
void aplicarOperacionesMorfologicas(const Mat& imagen, int tamanoKernel, const string& nombreImagen, int indice) {
    Mat erosionada, dilatada, topHat, blackHat, resultado;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(tamanoKernel, tamanoKernel));

    // Erosión
    erode(imagen, erosionada, kernel);
    agregarTexto(erosionada, "Erosion", Point(10, 30));

    // Dilatación
    dilate(imagen, dilatada, kernel);
    agregarTexto(dilatada, "Dilatacion", Point(10, 30));

    // Top Hat
    morphologyEx(imagen, topHat, MORPH_TOPHAT, kernel);
    agregarTexto(topHat, "Top Hat", Point(10, 30));

    // Black Hat
    morphologyEx(imagen, blackHat, MORPH_BLACKHAT, kernel);
    agregarTexto(blackHat, "Black Hat", Point(10, 30));

    // Original + (Top Hat - Black Hat)
    resultado = imagen + (topHat - blackHat);
    agregarTexto(resultado, "Resultado", Point(10, 30));

    // Concatenar imágenes en una sola ventana
    Mat combinada;
    hconcat(vector<Mat>{imagen, erosionada, dilatada}, combinada);
    Mat combinada2;
    hconcat(vector<Mat>{topHat, blackHat, resultado}, combinada2);
    vconcat(vector<Mat>{combinada, combinada2}, combinada);

    // Mostrar resultados
    string nombreVentana = nombreImagen + " - Tamaño de Kernel " + to_string(tamanoKernel) + " (" + to_string(indice) + ")";
    namedWindow(nombreVentana, WINDOW_NORMAL);
    imshow(nombreVentana, combinada);
}

int main() {
    // Cargar imágenes médicas en escala de grises
    Mat imagen1 = imread("imagen1.jpg", IMREAD_GRAYSCALE);
    Mat imagen2 = imread("imagen2.jpg", IMREAD_GRAYSCALE);
    Mat imagen3 = imread("imagen3.jpg", IMREAD_GRAYSCALE);

    if (imagen1.empty() || imagen2.empty() || imagen3.empty()) {
        cerr << "Error al cargar imágenes" << endl;
        return -1;
    }

    // Aplicar operaciones morfológicas con diferentes tamaños de kernel
    vector<int> tamanosKernel = {15, 25, 37};

    for (int tamano : tamanosKernel) {
        cout << "Tamaño del kernel: " << tamano << endl;
        aplicarOperacionesMorfologicas(imagen1, tamano, "Imagen 1", tamano);
        aplicarOperacionesMorfologicas(imagen2, tamano, "Imagen 2", tamano);
        aplicarOperacionesMorfologicas(imagen3, tamano, "Imagen 3", tamano);
    }

    // Esperar una tecla antes de salir
    waitKey(0);

    return 0;
}
