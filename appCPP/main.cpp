#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <array>
#include <filesystem>
#include "../Trainer/include/Model/FaceNet.h"

using namespace cv;
static torch::Device g_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
static const std::array<float, 3> kImageMean = {0.485f, 0.456f, 0.406f};
static const std::array<float, 3> kImageStd  = {0.229f, 0.224f, 0.225f};

namespace paths {
    const std::string embeds_file = "embeddings.pt";
    const std::string names_file  = "names.pt";
    const std::string haarcascade = "haarcascade_frontalface_alt.xml";
}

std::filesystem::path get_repo_root()
{
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::filesystem::path get_model_path()
{
    const auto repo_root = get_repo_root();
    std::filesystem::path candidates[] = {
        repo_root / "models" / "model.pt",
        std::filesystem::current_path() / "models" / "model.pt",
        repo_root / "build" / "Release" / "model.pt"
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return candidates[0];
}

int64_t count_dataset_classes(const std::filesystem::path& root)
{
    if (!std::filesystem::exists(root)) {
        return 1;
    }

    int64_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (entry.is_directory()) {
            ++count;
        }
    }

    return count > 0 ? count : 1;
}

bool load_face_cascade(CascadeClassifier& cascade, std::string& log)
{
    const std::vector<std::string> candidates = {
        paths::haarcascade,
        (get_repo_root() / "models" / paths::haarcascade).string(),
        (get_repo_root() / "data" / paths::haarcascade).string(),
        "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml",
        "C:/libs/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml"
    };

    for (const auto& candidate : candidates) {
        if (cascade.load(candidate)) {
            return true;
        }
        log += "Failed to load cascade from: " + candidate + "\n";
    }

    return false;
}

class Faces
{
private:
    std::string embeds_path;
    std::string names_path;
    torch::Tensor Embeds;
    std::vector<std::string> names;

public:
    Faces(std::string embeds_file, std::string names_file)
        : embeds_path(embeds_file), names_path(names_file)
    {
        load_embeds();
        load_names();
    }

    void load_embeds()
    {
        if (std::filesystem::exists(embeds_path)) {
            try {
                torch::load(Embeds, embeds_path);
            } catch (const c10::Error& e) {
                std::cerr << "Failed to load embeddings (" << e.what()
                          << "), starting fresh.\n";
                Embeds = torch::zeros({0, 128}, torch::kFloat32);
            }
        } else {
            Embeds = torch::zeros({0, 128}, torch::kFloat32);
        }
    }

    void save_embeds()
    {
        torch::save(Embeds, embeds_path);
    }

    void load_names()
    {
        names.clear();
        std::ifstream file(names_path);
        if (!file.is_open()) return;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) names.push_back(line);
        }
    }

    void save_names()
    {
        std::ofstream file(names_path, std::ios::trunc);
        for (const auto& s : names) {
            file << s << '\n';
        }
    }

    void add_Face(const torch::Tensor& new_emb, const std::string& name)
    {
        torch::Tensor emb_to_add = new_emb.detach().to(torch::kCPU).to(torch::kFloat32);

        if (emb_to_add.dim() == 1) {
            emb_to_add = emb_to_add.unsqueeze(0);
        }

        Embeds = torch::cat({Embeds, emb_to_add}, 0);
        names.push_back(name);

        save_embeds();
        save_names();
    }

    std::string search_faces(const torch::Tensor& searchembed, float threshold , float& confidence)
    {
        if (Embeds.size(0) == 0) return "UNKNOWN";

        torch::Tensor query = searchembed.detach().to(torch::kCPU).to(torch::kFloat32);
        if (query.dim() == 2) query = query.squeeze(0); 

        float best_dist = std::numeric_limits<float>::max();
        int best_idx = -1;

        for (int i = 0; i < Embeds.size(0); i++) {
            torch::Tensor dist = torch::norm(Embeds[i] - query, 2);
            float d = dist.item<float>();
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }
        confidence = best_dist;
        if (best_idx >= 0 && best_dist < threshold) {
            return names[best_idx];
        }
        return "UNKNOWN";
    }

    size_t face_count() const { return names.size(); }
};


VideoCapture open_webcam()
{
    VideoCapture cap;
    int deviceID = 0;
    int apiID = CAP_ANY;

    cap.open(deviceID, apiID);
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        exit(-1);
    }
    return cap;
}
static torch::Tensor g_lastEmbedding;
static bool g_lastFaceValid = false;

void process_frame(Mat& frame, CascadeClassifier& face_cascade,
                    model::FaceNet& model, Faces& face_processor)
{
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces);

    g_lastFaceValid = false;

    for (const auto& face : faces) {
        Mat faceROI = frame(face).clone();
        cvtColor(faceROI, faceROI, COLOR_BGR2RGB);
        resize(faceROI, faceROI, Size(112, 112));
        faceROI.convertTo(faceROI, CV_32F, 1.0 / 255.0);

        auto tensor = torch::from_blob(faceROI.data, {faceROI.rows, faceROI.cols, 3}, torch::kFloat32);
        tensor = tensor.permute({2, 0, 1}).clone().unsqueeze(0);
        auto normalize = torch::data::transforms::Normalize<>(
            std::vector<double>{0.485, 0.456, 0.406},
            std::vector<double>{0.229, 0.224, 0.225}
        );
        tensor = normalize(tensor);
        tensor = tensor.to(g_device);

        torch::Tensor embed_ = model->embed(tensor);
        float confidence = 0.0;
        std::string name = face_processor.search_faces(embed_ ,0.7 , confidence);


        g_lastEmbedding = embed_.detach().to(torch::kCPU).clone();
        g_lastFaceValid = true;

        rectangle(frame, face, Scalar(0, 0, 255), 3);
        int fontFace = FONT_HERSHEY_SIMPLEX;
        Scalar color(0, 0, 255);
        Point textOrg(face.x, face.y - 10);
        putText(frame, name, textOrg, fontFace, 0.7, color, 2);
        Scalar color_(0, 0, 255);
        Point textOrg_(face.x, face.y -25);
        putText(frame, std::to_string(confidence), textOrg_, fontFace, 0.7, color_, 2);
    }
}

static std::string g_modelLog;
static ID3D11Device*            g_pd3dDevice            = nullptr;
static ID3D11DeviceContext*     g_pd3dDeviceContext     = nullptr;
static IDXGISwapChain*          g_pSwapChain            = nullptr;
static ID3D11RenderTargetView*  g_mainRenderTargetView  = nullptr;
static bool                     g_SwapChainOccluded     = false;
static UINT                     g_ResizeWidth           = 0, g_ResizeHeight = 0;
static ID3D11Texture2D*         g_pWebcamTexture        = nullptr;
static ID3D11ShaderResourceView* g_pWebcamSRV           = nullptr;
static int                       g_webcamTexW            = 0, g_webcamTexH = 0;

void settings_texture(Mat& frame_)
{
    bool needsRecreate = (g_pWebcamTexture == nullptr)
        || (g_webcamTexW != frame_.cols)
        || (g_webcamTexH != frame_.rows);

    if (!needsRecreate) return;

    if (g_pWebcamSRV)     { g_pWebcamSRV->Release();     g_pWebcamSRV     = nullptr; }
    if (g_pWebcamTexture) { g_pWebcamTexture->Release(); g_pWebcamTexture = nullptr; }

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width          = (UINT)frame_.cols;
    desc.Height         = (UINT)frame_.rows;
    desc.MipLevels      = 1;
    desc.ArraySize      = 1;
    desc.Format         = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage          = D3D11_USAGE_DYNAMIC;
    desc.BindFlags      = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    HRESULT hr = g_pd3dDevice->CreateTexture2D(&desc, nullptr, &g_pWebcamTexture);
    if (FAILED(hr)) {
        std::cerr << "CreateTexture2D failed: 0x" << std::hex << hr << std::endl;
        return;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format                    = desc.Format;
    srvDesc.ViewDimension             = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels       = 1;
    srvDesc.Texture2D.MostDetailedMip = 0;

    hr = g_pd3dDevice->CreateShaderResourceView(g_pWebcamTexture, &srvDesc, &g_pWebcamSRV);
    if (FAILED(hr)) {
        std::cerr << "CreateShaderResourceView failed: 0x" << std::hex << hr << std::endl;
        g_pWebcamTexture->Release();
        g_pWebcamTexture = nullptr;
        return;
    }

    g_webcamTexW = frame_.cols;
    g_webcamTexH = frame_.rows;
}

void settings_mapped_subresource(Mat& rgba_, Mat& frame_)
{
    if (!g_pWebcamTexture) return;

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    HRESULT hr = g_pd3dDeviceContext->Map(g_pWebcamTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    if (SUCCEEDED(hr)) {
        BYTE* pDest       = (BYTE*)mappedResource.pData;
        const BYTE* pSrc  = rgba_.data;
        int rowPitch      = mappedResource.RowPitch;
        int srcStep       = (int)rgba_.step;
        int widthBytes    = frame_.cols * 4;

        for (int y = 0; y < frame_.rows; y++) {
            memcpy(pDest + y * rowPitch, pSrc + y * srcStep, widthBytes);
        }

        g_pd3dDeviceContext->Unmap(g_pWebcamTexture, 0);
    } else {
        std::cerr << "Map failed: 0x" << std::hex << hr << std::endl;
    }
}

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    ImGui_ImplWin32_EnableDpiAwareness();
    float main_scale = 1.0f;

    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"ImGui Window", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"Dear ImGui", WS_OVERLAPPEDWINDOW, 100, 100, (int)(1280 * main_scale), (int)(800 * main_scale), nullptr, nullptr, wc.hInstance, nullptr);

    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    bool done = false;

    VideoCapture cap = open_webcam();
    Mat frame;

    cap >> frame;
    if (frame.empty()) {
        std::cerr << "ERROR! Camera opened but returned empty frame\n";
        ImGui_ImplDX11_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
        CleanupDeviceD3D();
        ::DestroyWindow(hwnd);
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    for (int i = 0; i < 5; i++) cap >> frame;

    CascadeClassifier face_cascade;
    if (!load_face_cascade(face_cascade, g_modelLog)) {
        g_modelLog += "Using default face detector fallback.\n";
    }

    Faces face_processor(paths::embeds_file, paths::names_file);

    const int64_t embedding_dim = 128;
    model::FaceNet model(11257, embedding_dim, model::LossType::TripletSemiHard, 64.0, 0.5, true);
    const auto model_path = get_model_path();
    g_modelLog += "Loading FaceNet model from: " + model_path.string() + "\n";
    bool model_loaded = false;
    try {
        torch::load(model, model_path.string());
        g_modelLog += "FaceNet model loaded successfully.\n";
        model_loaded = true;
    } catch (const c10::Error& e) {
        g_modelLog += "Error loading FaceNet model: " + std::string(e.what()) + "\n";
    }
    model->to(g_device);
    model->eval();

    char nameInputBuf[128] = "";
    std::string registerStatus;

    while (!done) {
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done) break;

        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED) {
            ::Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        process_frame(frame, face_cascade, model, face_processor);
        settings_texture(frame);
        cv::Mat rgba;
        cv::cvtColor(frame, rgba, cv::COLOR_BGR2RGBA);
        settings_mapped_subresource(rgba, frame);

        ImGui::Begin("Webcam");
        if (g_pWebcamSRV) {
            ImGui::Image((ImTextureID)(intptr_t)g_pWebcamSRV, ImVec2(1000, 500));
        } else {
            ImGui::Text("Webcam texture not ready!");
        }
        ImGui::End();

        ImGui::Begin("Face Registration");
        ImGui::Text("Known faces: %zu", face_processor.face_count());
        ImGui::InputText("Name", nameInputBuf, sizeof(nameInputBuf));

        bool canRegister = g_lastFaceValid && nameInputBuf[0] != '\0';
        if (!canRegister) ImGui::BeginDisabled();
        if (ImGui::Button("Register Face")) {
            face_processor.add_Face(g_lastEmbedding, nameInputBuf);
            registerStatus = std::string("Registered '") + nameInputBuf + "'";
            nameInputBuf[0] = '\0';
        }
        if (!canRegister) ImGui::EndDisabled();

        if (!g_lastFaceValid) {
            ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "No face currently detected.");
        }
        if (!registerStatus.empty()) {
            ImGui::TextColored(ImVec4(0.3f, 1, 0.3f, 1), "%s", registerStatus.c_str());
        }
        ImGui::End();

        ImGui::Begin("Model Log");
        ImGui::Text(model_loaded ? "Model is loaded." : "Model is NOT loaded.");
        ImGui::Separator();
        ImGui::TextWrapped("Logs:");
        ImGui::BeginChild("ModelLogChild", ImVec2(0, 150), true, ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::TextUnformatted(g_modelLog.c_str());
        ImGui::EndChild();
        ImGui::End();

        ImGui::Render();
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, (float*)&clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        HRESULT presentResult = g_pSwapChain->Present(1, 0);
        g_SwapChainOccluded = (presentResult == DXGI_STATUS_OCCLUDED);
        if (presentResult == DXGI_ERROR_DEVICE_REMOVED || presentResult == DXGI_ERROR_DEVICE_RESET) {
            std::cerr << "D3D device lost!\n";
            break;
        }
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

bool CreateDeviceD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount          = 2;
    sd.BufferDesc.Format    = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage          = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow         = hWnd;
    sd.SampleDesc.Count     = 1;
    sd.Windowed             = TRUE;
    sd.SwapEffect           = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED)
        res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pWebcamSRV)     { g_pWebcamSRV->Release();     g_pWebcamSRV     = nullptr; }
    if (g_pWebcamTexture) { g_pWebcamTexture->Release(); g_pWebcamTexture = nullptr; }
    if (g_pSwapChain)     { g_pSwapChain->Release();     g_pSwapChain     = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice)     { g_pd3dDevice->Release();     g_pd3dDevice     = nullptr; }
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED) return 0;
        g_ResizeWidth  = (UINT)LOWORD(lParam);
        g_ResizeHeight = (UINT)HIWORD(lParam);
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}