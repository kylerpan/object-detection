#include "engine.h"
#include "postprocess.h"
#include <opencv2/opencv.hpp>
#include <chrono>

using ms = std::chrono::milliseconds;
using us = std::chrono::microseconds;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline void sleep(int32_t n) { std::this_thread::sleep_for(ms(n)); }
inline Time get_time() { return std::chrono::high_resolution_clock::now(); }
inline uint32_t get_ms(Time& t0, Time& t1)
{
    auto t = std::chrono::duration_cast<ms>(t1 - t0);
    return (uint32_t)t.count();
}
inline uint32_t get_us(Time& t0, Time& t1)
{
    auto t = std::chrono::duration_cast<us>(t1 - t0);
    return (uint32_t)t.count();
}

int main() {
    Options options;
    //options.optBatchSizes = {1};

    Engine engine(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    const std::string onnxModelpath = "C:/Kyle_Work/models/yolov3.onnx";

    // building engine
    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }
    
    // loading engine
    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    auto tensors = engine.getTensors({ "image_input", "Identity", "Identity_1", "Identity_2" });
    if (tensors.size() == 0)
    {
        return -1;
    }
    int32_t net_h = tensors[0].h;
    int32_t net_w = tensors[0].w;
    int32_t net_c = tensors[0].c;
    float* net_data = tensors[0].data;
    assert(net_c == 3);
    assert(tensors.size() == 4);

    // output tensor with NHWC
    std::vector<Tensor> out_tensors(3);
    std::vector<float> buffer(tensors[1].num_elems + tensors[2].num_elems + tensors[3].num_elems);
    float* p = buffer.data();
    for (int i = 0; i < 3; i++)
    {
        out_tensors[i] = tensors[i + 1];
        out_tensors[i].data = p;
        p += out_tensors[i].num_elems;
    }

    // open camera
    cv::Mat frame;
    cv::namedWindow("Display window");
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cout << "Unable to open camera" << std::endl;
    }

    // get the padding of video
    int frame_height = (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_width = (int) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    bool taller = frame_height > frame_width;
    int padding = taller ? frame_height - frame_width : frame_width - frame_height;
    printf("Video dimensions = (%d, %d, %d)\n", frame_height, frame_width, padding);

    int i = 0;
    while (true)
    {
        auto t0 = get_time();
        cap >> frame;
        cv::flip(frame, frame, 1);
        cv::Mat resized_frame(net_h, net_w, CV_8UC3);

        // pad frame to square, the resize
        auto t1 = get_time();
        if (taller)
        {
            cv::copyMakeBorder(frame, resized_frame, 0, 0, 0, padding, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        else
        {
            cv::copyMakeBorder(frame, resized_frame, 0, padding, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        };

        if (i == 0) {
            printf("Padded frame shape = (%d, %d, %d)\n", resized_frame.rows, resized_frame.cols, resized_frame.channels());
        }

        cv::resize(resized_frame, resized_frame, cv::Size(net_w, net_h), 0.0, 0.0, cv::INTER_LINEAR);

        // normalizing frame
        cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

        uint8_t* p_src = resized_frame.data;
        int frame_size = net_w * net_h;
        for (int i = 0; i < frame_size; i++)
        {
            for (int c = 0; c < net_c; c++)
            {
                net_data[c * frame_size + i] = *p_src++ / 255.0f;
            }
        }

        // reference
        auto t2 = get_time();
        engine.execute();
        //printf("Executed Engine\n");

        // change from NCHW to NHWC
        auto t3 = get_time();
        for (int i = 0; i < 3; i++)
        {
            float* psrc = tensors[i + 1].data;
            float* pdst = out_tensors[i].data;
            int frame_size = out_tensors[i].h * out_tensors[i].w;
            int num_chan = out_tensors[i].c;
            for (int c = 0; c < num_chan; c++)
            {
                for (int f = 0; f < frame_size; f++)
                {
                    pdst[f * num_chan + c] = *psrc++;
                }
            }
        }

        // postprocess
        std::vector<BBox> boxes;
        Postprocess(frame_height, frame_width, net_h, net_w, out_tensors, boxes);
        auto t4 = get_time();
        //printf("Got all Boxes\n");
        if ((i & 31) == 0) printf("time get_video=%dus, preproc=%dus, infer=%dus, postproc=%dus\n", get_us(t0, t1), get_us(t1, t2), get_us(t2, t3), get_us(t3, t4));

        // draw bounding box
        DrawBox(boxes, frame);
        //printf("Draw Boxes\n");

        // display
        cv::imshow("Display window", frame);

        char k = cv::waitKey(1);
        if (k == 'p')
        {
            k = cv::waitKey(-1);
        }
        if (k == 'q')
        {
            break;
        }

        i++;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
