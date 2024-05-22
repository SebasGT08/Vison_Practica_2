#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

using namespace cv;
using namespace std;

// Función para agregar ruido de sal y pimienta a una imagen
void agregarRuidoSalPimienta(Mat &imagen, float prob_sal, float prob_pimienta) {
    RNG rng;
    for (int i = 0; i < imagen.rows; i++) {
        for (int j = 0; j < imagen.cols; j++) {
            float valor_aleatorio = rng.uniform(0.0f, 1.0f);
            if (valor_aleatorio < prob_sal) {
                imagen.at<Vec3b>(i, j) = Vec3b(255, 255, 255); // Sal (blanco)
            } else if (valor_aleatorio < prob_sal + prob_pimienta) {
                imagen.at<Vec3b>(i, j) = Vec3b(0, 0, 0); // Pimienta (negro)
            }
        }
    }
}

// Variables globales para los trackbars
int deslizador_sal = 0; // Deslizador para la probabilidad de sal
int deslizador_pimienta = 0; // Deslizador para la probabilidad de pimienta
int deslizador_tamano_mascara = 1; // Deslizador para el tamaño de la máscara de los filtros
Mat imagen_original, imagen_con_ruido;
Size tamano_nuevo(400, 240); // Tamaño nuevo para redimensionar el video

// Función callback para los trackbars
void on_trackbar(int, void*) {
    imagen_con_ruido = imagen_original.clone();
    float prob_sal = deslizador_sal / 100.0;
    float prob_pimienta = deslizador_pimienta / 100.0;
    agregarRuidoSalPimienta(imagen_con_ruido, prob_sal, prob_pimienta);
    imshow("Video con Ruido", imagen_con_ruido);
}

// Función para aplicar filtros de suavizado y devolver los resultados
void aplicarFiltros(const Mat &imagen, Mat &filtrado_mediana, Mat &filtrado_blur, Mat &filtrado_gaussiano) {
    int tamano_mascara = deslizador_tamano_mascara * 2 + 1; // Tamaño de la máscara (debe ser impar)
    
    // Aplicar filtro de mediana
    medianBlur(imagen, filtrado_mediana, tamano_mascara);
    // Aplicar filtro de blur (promedio)
    blur(imagen, filtrado_blur, Size(tamano_mascara, tamano_mascara));
    // Aplicar filtro Gaussiano
    GaussianBlur(imagen, filtrado_gaussiano, Size(tamano_mascara, tamano_mascara), 1.5);
}

// Función para detección de bordes usando Canny
void deteccionBordes(const Mat &imagen, Mat &bordes) {
    Mat imagen_gris;
    cvtColor(imagen, imagen_gris, COLOR_BGR2GRAY);
    Canny(imagen_gris, bordes, 50, 150);
    cvtColor(bordes, bordes, COLOR_GRAY2BGR); // Convertir a BGR para que coincida el tipo
}

// Función para detección de bordes usando el algoritmo Sobel
void deteccionBordesSobel(const Mat &imagen, Mat &bordes) {
    Mat imagen_gris, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    cvtColor(imagen, imagen_gris, COLOR_BGR2GRAY);
    
    Sobel(imagen_gris, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(imagen_gris, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, bordes);
    cvtColor(bordes, bordes, COLOR_GRAY2BGR); // Convertir a BGR para que coincida el tipo
}

// Función para agregar texto a una imagen
void agregarTexto(Mat &imagen, const string &texto, const Point &posicion) {
    putText(imagen, texto, posicion, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
}

int main(int argc, char** argv) {
    // Cargar el video desde el archivo
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cout << "Error al abrir el video" << endl;
        return -1;
    }

    // Crear ventanas y trackbars
    namedWindow("Video con Ruido", 1);
    createTrackbar("Sal", "Video con Ruido", &deslizador_sal, 100, on_trackbar);
    createTrackbar("Pimienta", "Video con Ruido", &deslizador_pimienta, 100, on_trackbar);
    createTrackbar("Tamano de la Mascara", "Video con Ruido", &deslizador_tamano_mascara, 10, on_trackbar);

    namedWindow("Deteccion de Bordes Canny", 1);
    namedWindow("Deteccion de Bordes Sobel", 1);

    while (true) {
        cap >> imagen_original; // Capturar un fotograma del video
        if (imagen_original.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0); // Reiniciar el video si llega al final
            cap >> imagen_original;
            if (imagen_original.empty()) break; // Salir si no se puede capturar un fotograma
        }

        resize(imagen_original, imagen_original, tamano_nuevo); // Redimensionar el fotograma

        on_trackbar(0, 0); // Aplicar ruido de sal y pimienta

        // Aplicar filtros de suavizado
        Mat filtrado_mediana, filtrado_blur, filtrado_gaussiano;
        aplicarFiltros(imagen_con_ruido, filtrado_mediana, filtrado_blur, filtrado_gaussiano);

        // Detectar bordes en las imágenes filtradas usando Canny
        Mat bordes_mediana, bordes_blur, bordes_gaussiano;
        deteccionBordes(filtrado_mediana, bordes_mediana);
        deteccionBordes(filtrado_blur, bordes_blur);
        deteccionBordes(filtrado_gaussiano, bordes_gaussiano);

        // Detectar bordes en las imágenes filtradas usando Sobel
        Mat bordes_mediana_sobel, bordes_blur_sobel, bordes_gaussiano_sobel;
        deteccionBordesSobel(filtrado_mediana, bordes_mediana_sobel);
        deteccionBordesSobel(filtrado_blur, bordes_blur_sobel);
        deteccionBordesSobel(filtrado_gaussiano, bordes_gaussiano_sobel);

        // Crear una imagen grande para mostrar todas las imágenes filtradas y sus bordes (Canny)
        Mat resultado_filtrado(Size(imagen_con_ruido.cols * 2, imagen_con_ruido.rows * 2), imagen_con_ruido.type());
        Mat resultado_bordes_canny(Size(imagen_con_ruido.cols * 2, imagen_con_ruido.rows * 2), imagen_con_ruido.type());

        // Copiar las imágenes filtradas a la imagen de resultado y agregar texto (Canny)
        filtrado_mediana.copyTo(resultado_filtrado(Rect(0, 0, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_filtrado, "Mediana", Point(10, 30));

        filtrado_blur.copyTo(resultado_filtrado(Rect(imagen_con_ruido.cols, 0, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_filtrado, "Blur", Point(imagen_con_ruido.cols + 10, 30));

        filtrado_gaussiano.copyTo(resultado_filtrado(Rect(0, imagen_con_ruido.rows, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_filtrado, "Gaussiano", Point(10, imagen_con_ruido.rows + 30));

        imagen_con_ruido.copyTo(resultado_filtrado(Rect(imagen_con_ruido.cols, imagen_con_ruido.rows, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_filtrado, "Original", Point(imagen_con_ruido.cols + 10, imagen_con_ruido.rows + 30));

        // Copiar las imágenes de bordes a la imagen de resultado y agregar texto (Canny)
        bordes_mediana.copyTo(resultado_bordes_canny(Rect(0, 0, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_bordes_canny, "Bordes Mediana", Point(10, 30));

        bordes_blur.copyTo(resultado_bordes_canny(Rect(imagen_con_ruido.cols, 0, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_bordes_canny, "Bordes Blur", Point(imagen_con_ruido.cols + 10, 30));

        bordes_gaussiano.copyTo(resultado_bordes_canny(Rect(0, imagen_con_ruido.rows, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_bordes_canny, "Bordes Gaussiano", Point(10, imagen_con_ruido.rows + 30));

        deteccionBordes(imagen_con_ruido, imagen_con_ruido); // Detectar bordes en la imagen con ruido
        imagen_con_ruido.copyTo(resultado_bordes_canny(Rect(imagen_con_ruido.cols, imagen_con_ruido.rows, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_bordes_canny, "Bordes Original", Point(imagen_con_ruido.cols + 10, imagen_con_ruido.rows + 30));

        // Crear una imagen grande para mostrar todas las imágenes filtradas y sus bordes (Sobel)
        Mat resultado_filtrado_sobel(Size(imagen_con_ruido.cols * 2, imagen_con_ruido.rows * 2), imagen_con_ruido.type());
        Mat resultado_bordes_sobel(Size(imagen_con_ruido.cols * 2, imagen_con_ruido.rows * 2), imagen_con_ruido.type());

        // Copiar las imágenes filtradas a la imagen de resultado y agregar texto (Sobel)
        filtrado_mediana.copyTo(resultado_filtrado_sobel(Rect(0, 0, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_filtrado_sobel, "Mediana", Point(10, 30));

        filtrado_blur.copyTo(resultado_filtrado_sobel(Rect(imagen_con_ruido.cols, 0, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_filtrado_sobel, "Blur", Point(imagen_con_ruido.cols + 10, 30));

        filtrado_gaussiano.copyTo(resultado_filtrado_sobel(Rect(0, imagen_con_ruido.rows, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_filtrado_sobel, "Gaussiano", Point(10, imagen_con_ruido.rows + 30));

        imagen_con_ruido.copyTo(resultado_filtrado_sobel(Rect(imagen_con_ruido.cols, imagen_con_ruido.rows, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_filtrado_sobel, "Original", Point(imagen_con_ruido.cols + 10, imagen_con_ruido.rows + 30));

        // Copiar las imágenes de bordes a la imagen de resultado y agregar texto (Sobel)
        bordes_mediana_sobel.copyTo(resultado_bordes_sobel(Rect(0, 0, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_bordes_sobel, "Bordes Mediana", Point(10, 30));

        bordes_blur_sobel.copyTo(resultado_bordes_sobel(Rect(imagen_con_ruido.cols, 0, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_bordes_sobel, "Bordes Blur", Point(imagen_con_ruido.cols + 10, 30));

        bordes_gaussiano_sobel.copyTo(resultado_bordes_sobel(Rect(0, imagen_con_ruido.rows, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_bordes_sobel, "Bordes Gaussiano", Point(10, imagen_con_ruido.rows + 30));

        deteccionBordesSobel(imagen_con_ruido, imagen_con_ruido); // Detectar bordes en la imagen con ruido
        imagen_con_ruido.copyTo(resultado_bordes_sobel(Rect(imagen_con_ruido.cols, imagen_con_ruido.rows, imagen_con_ruido.cols, imagen_con_ruido.rows)));
        agregarTexto(resultado_bordes_sobel, "Bordes Original", Point(imagen_con_ruido.cols + 10, imagen_con_ruido.rows + 30));

        // Mostrar las imágenes de resultado
        imshow("Video Filtrado", resultado_filtrado);
        imshow("Deteccion de Bordes Canny", resultado_bordes_canny);

        imshow("Deteccion de Bordes Sobel", resultado_bordes_sobel);

        // Salir si se presiona la tecla ESC
        if (waitKey(23) == 27) break;
    }

    return 0;
}
