#include "../tests/test_model.cpp"

int main() {
    testFaceRecognitionProjector(3, 256, 0.5);
    testFaceRecognitionModel(3, 256, 0.5);
    return 0;
}