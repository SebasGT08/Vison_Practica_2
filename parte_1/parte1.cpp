#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <curl/curl.h>
#include <json/json.h>
#include <memory>
#include <array>
#include <stdexcept>
#include <cstdio>

using namespace std;
using namespace cv;

// Función para ejecutar un comando del sistema y capturar la salida
std::string ejecutarComando(const char* cmd) {
    std::array<char, 128> buffer;
    std::string resultado;
    std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() falló!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        resultado += buffer.data();
    }
    return resultado;
}

// Función para obtener la URL directa del stream de YouTube
std::string obtenerURLStreamYouTube(const std::string& youtubeUrl) {
    std::string comando = "youtube-dl -f best --get-url " + youtubeUrl;
    return ejecutarComando(comando.c_str());
}

// Variables globales para parámetros de los filtros
int gammaEntero = 10; // Valor entero para el parámetro gamma (inicializado en 10)
double gammaValor = 1.0; // Valor inicial para la corrección gamma

// Función de callback para los trackbars
void funcionGamma(int valor, void *) { gammaValor = valor / 10.0; }

// Función que aplica ecualización de histograma
Mat aplicarEcualizacionHistograma(Mat& frame) {
    Mat ycrcb;
    cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);
    vector<Mat> canales;
    split(ycrcb, canales);
    equalizeHist(canales[0], canales[0]);
    merge(canales, ycrcb);
    Mat resultado;
    cvtColor(ycrcb, resultado, COLOR_YCrCb2BGR);
    return resultado;
}

// Función que aplica el filtro CLAHE
Mat aplicarCLAHE(Mat& frame) {
    Mat lab;
    cvtColor(frame, lab, COLOR_BGR2Lab);
    vector<Mat> lab_planes(3);
    split(lab, lab_planes);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    Mat resultado;
    clahe->apply(lab_planes[0], resultado);
    resultado.copyTo(lab_planes[0]);
    merge(lab_planes, lab);
    Mat resultado_final;
    cvtColor(lab, resultado_final, COLOR_Lab2BGR);
    return resultado_final;
}

// Función que aplica la corrección gamma
Mat aplicarCorreccionGamma(Mat& frame, double gamma) {
    Mat resultado;
    Mat tablaLUT(1, 256, CV_8U);
    uchar* p = tablaLUT.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    LUT(frame, tablaLUT, resultado);
    return resultado;
}

// Función para mostrar los FPS en el frame
void mostrarFPS(Mat& frame, double fps) {
    string textoFPS = "FPS: " + to_string(int(fps));
    putText(frame, textoFPS, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
}

// Función para detectar movimiento
void detectarMovimiento(Mat& frameActual, Mat& frameAnterior, Mat& movimiento, Ptr<BackgroundSubtractor> pBackSub) {
    Mat resta;
    absdiff(frameActual, frameAnterior, resta);
    threshold(resta, movimiento, 10, 255, THRESH_BINARY);
    pBackSub->apply(frameActual, movimiento);
    frameAnterior = frameActual.clone();
}

int main(int argc, char* args[]) {
    // URL del stream de video en vivo de YouTube
    string youtubeUrl = "https://www.youtube.com/watch?v=tWu34gp3Rmk";
    // Obtener el enlace directo del video en vivo de YouTube
    string streamUrl;
    try {
        streamUrl = obtenerURLStreamYouTube(youtubeUrl);
    } catch (const std::exception& e) {
        cerr << "Error al obtener la URL del stream de YouTube: " << e.what() << endl;
        return -1;
    }

    if (streamUrl.empty()) {
        cerr << "Error al obtener la URL del stream de YouTube!" << endl;
        return -1;
    }

    // Manejo de Video
    VideoCapture video(streamUrl);

    // Verificamos si la cámara se pudo abrir
    if (!video.isOpened()) {
        cerr << "Error al abrir el stream de video!" << endl;
        return -1;
    }

    // Crear ventanas para mostrar los resultados
    namedWindow("Original y Movimiento", WINDOW_AUTOSIZE);
    namedWindow("Histograma Ecualizado y Movimiento", WINDOW_AUTOSIZE);
    namedWindow("CLAHE y Movimiento", WINDOW_AUTOSIZE);
    namedWindow("Correccion Gamma y Movimiento", WINDOW_AUTOSIZE);

    // Crear trackbars para ajustar parámetros en la ventana "Corrección Gamma y Movimiento"
    createTrackbar("Gamma", "Correccion Gamma y Movimiento", &gammaEntero, 50, funcionGamma);

    Mat frameColor, frameAnteriorColor, movimientoOriginal;
    Mat frameAnteriorHistograma, movimientoHistograma;
    Mat frameAnteriorCLAHE, movimientoCLAHE;
    Mat frameAnteriorGamma, movimientoGamma;

    // Crear objetos para substracción de fondo
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractor> pBackSubHistograma = createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractor> pBackSubCLAHE = createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractor> pBackSubGamma = createBackgroundSubtractorMOG2();
    
    // Variables para calcular FPS
    double fps = 0.0;
    int conteoFrames = 0;
    double inicio = getTickCount();

    // Bucle principal
    while (true) {
        video >> frameColor; // Capturar frame del video
        if (frameColor.empty()) break; // Salir si no hay más frames

        // Redimensionar el frameColor una sola vez
        Size tamanoPequeno(800, 600); // Cambiar este tamaño según sea necesario
        resize(frameColor, frameColor, tamanoPequeno);

        // Mostrar los FPS
        conteoFrames++;
        double actual = (getTickCount() - inicio) / getTickFrequency();
        fps = conteoFrames / actual;

        mostrarFPS(frameColor, fps);

        // Convertir a escala de grises para movimiento
        Mat frameGris;
        cvtColor(frameColor, frameGris, COLOR_BGR2GRAY);

        if (frameAnteriorColor.empty()) {
            frameAnteriorColor = frameGris.clone(); // Inicializar el primer frame
            cout << "Inicializa el primer frame ..." << endl;
        }

        // Detectar movimiento en el frame original
        detectarMovimiento(frameGris, frameAnteriorColor, movimientoOriginal, pBackSub);

        // Aplicar filtros adicionales a la imagen en color
        Mat histogramaEcualizado = aplicarEcualizacionHistograma(frameColor);
        Mat resultadoCLAHE = aplicarCLAHE(frameColor);
        Mat correccionGamma = aplicarCorreccionGamma(frameColor, gammaValor);

        // Convertir a escala de grises para detección de movimiento en filtros
        Mat histogramaGris, claheGris, gammaGris;
        cvtColor(histogramaEcualizado, histogramaGris, COLOR_BGR2GRAY);
        cvtColor(resultadoCLAHE, claheGris, COLOR_BGR2GRAY);
        cvtColor(correccionGamma, gammaGris, COLOR_BGR2GRAY);

        // Detectar movimiento en los frames con filtros aplicados
        if (frameAnteriorHistograma.empty()) frameAnteriorHistograma = histogramaGris.clone();
        if (frameAnteriorCLAHE.empty()) frameAnteriorCLAHE = claheGris.clone();
        if (frameAnteriorGamma.empty()) frameAnteriorGamma = gammaGris.clone();

        detectarMovimiento(histogramaGris, frameAnteriorHistograma, movimientoHistograma, pBackSubHistograma);
        detectarMovimiento(claheGris, frameAnteriorCLAHE, movimientoCLAHE, pBackSubCLAHE);
        detectarMovimiento(gammaGris, frameAnteriorGamma, movimientoGamma, pBackSubGamma);

        // Crear una imagen combinada para cada filtro y su movimiento
        Mat combinadaOriginal(frameColor.rows, frameColor.cols * 2, frameColor.type());
        Mat combinadaHistograma(histogramaEcualizado.rows, histogramaEcualizado.cols * 2, histogramaEcualizado.type());
        Mat combinadaCLAHE(resultadoCLAHE.rows, resultadoCLAHE.cols * 2, resultadoCLAHE.type());
        Mat combinadaGamma(correccionGamma.rows, correccionGamma.cols * 2, correccionGamma.type());

        // Copiar las imágenes originales y de movimiento a las combinadas
        frameColor.copyTo(combinadaOriginal(Rect(0, 0, frameColor.cols, frameColor.rows)));
        cvtColor(movimientoOriginal, movimientoOriginal, COLOR_GRAY2BGR); // Convertir a BGR para que coincida el tipo
        movimientoOriginal.copyTo(combinadaOriginal(Rect(frameColor.cols, 0, movimientoOriginal.cols, movimientoOriginal.rows)));

        histogramaEcualizado.copyTo(combinadaHistograma(Rect(0, 0, histogramaEcualizado.cols, histogramaEcualizado.rows)));
        cvtColor(movimientoHistograma, movimientoHistograma, COLOR_GRAY2BGR);
        movimientoHistograma.copyTo(combinadaHistograma(Rect(histogramaEcualizado.cols, 0, movimientoHistograma.cols, movimientoHistograma.rows)));

        resultadoCLAHE.copyTo(combinadaCLAHE(Rect(0, 0, resultadoCLAHE.cols, resultadoCLAHE.rows)));
        cvtColor(movimientoCLAHE, movimientoCLAHE, COLOR_GRAY2BGR);
        movimientoCLAHE.copyTo(combinadaCLAHE(Rect(resultadoCLAHE.cols, 0, movimientoCLAHE.cols, movimientoCLAHE.rows)));

        correccionGamma.copyTo(combinadaGamma(Rect(0, 0, correccionGamma.cols, correccionGamma.rows)));
        cvtColor(movimientoGamma, movimientoGamma, COLOR_GRAY2BGR);
        movimientoGamma.copyTo(combinadaGamma(Rect(correccionGamma.cols, 0, movimientoGamma.cols, movimientoGamma.rows)));

        // Mostrar imágenes combinadas
        if (!combinadaOriginal.empty()) {
            imshow("Original y Movimiento", combinadaOriginal);
        }
        if (!combinadaHistograma.empty()) {
            imshow("Histograma Ecualizado y Movimiento", combinadaHistograma);
        }
        if (!combinadaCLAHE.empty()) {
            imshow("CLAHE y Movimiento", combinadaCLAHE);
        }
        if (!combinadaGamma.empty()) {
            imshow("Correccion Gamma y Movimiento", combinadaGamma);
        }

        // Salir si se presiona la tecla ESC
        if (waitKey(23) == 27) break;
    }

    // Liberar el video y destruir todas las ventanas
    video.release();
    destroyAllWindows();

    return 0;
}
