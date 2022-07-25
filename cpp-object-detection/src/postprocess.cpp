#include "postprocess.h"
#include "engine.h"
#include <string>
#include <sstream>
#include <iomanip>

int ANCHOR_NUM = 3;
int CLASS_NUM = 80;
std::vector<std::vector<int>> ANCHOR_WIDTH = { {116, 156, 373}, {30, 62, 59}, {10, 16, 33} };
std::vector<std::vector<int>> ANCHOR_HEIGHT = { {90, 198, 326}, {61, 45, 119}, {13, 30, 23} };
std::vector<std::string> LABEL = {
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
};

float sigmoid(float num)
{
    return 1 / (1 + std::expf(-num));
}

bool compareScore(BBox& a, BBox& b)
{
    return a.score > b.score;
}

float calcIOU(BBox& a, BBox& b)
{
    int x_left = std::max(a.x, b.x);
    int x_right = std::min(a.x + a.w, b.x + b.w);
    int y_top = std::max(a.y, b.y);
    int y_bottom = std::min(a.y + a.h, b.y + b.h);

    if (x_right < x_left || y_bottom < y_top)
    {
        return 0.0;
    }

    int intersection = (x_right - x_left) * (y_bottom - y_top);

    return (float)intersection / (a.area + b.area - intersection);
}


void Postprocess(int32_t img_h, int32_t img_w, int32_t net_h, int32_t net_w, std::vector<Tensor>& tensors, std::vector<BBox>& boxes, float score_threshold, float iou_threshold)
{
    // get all the bboxes
    std::vector<BBox> all_boxes;
    for (int t = 0; t < tensors.size(); t++)
    {
        float* tensor_data = tensors[t].data;
        for (int y = 0; y < tensors[t].h; y++)
        {
            for (int x = 0; x < tensors[t].w; x++)
            {
                for (int a = 0; a < ANCHOR_NUM; a++, tensor_data += CLASS_NUM + 5)
                {
                    float objectness_score = sigmoid(tensor_data[4]);
                    if (objectness_score < score_threshold)
                    {
                        continue;
                    }

                    float* class_scores = &tensor_data[5];
                    for (int c = 0; c < CLASS_NUM; c++)
                    {
                        float score = objectness_score * sigmoid(class_scores[c]);

                        if (score > score_threshold)
                        {
                            float xc = (sigmoid(tensor_data[0]) + x) / tensors[t].w;
                            float yc = (sigmoid(tensor_data[1]) + y) / tensors[t].h;
                            float w = std::expf(tensor_data[2]) * ANCHOR_WIDTH[t][a] / net_w;
                            float h = std::expf(tensor_data[3]) * ANCHOR_HEIGHT[t][a] / net_h;

                            float x0 = xc - w / 2;
                            float x1 = xc + w / 2;
                            float y0 = yc - h / 2;
                            float y1 = yc + h / 2;
                            x0 = std::max(0.0f, std::min(1.0f, x0));
                            x1 = std::max(0.0f, std::min(1.0f, x1));
                            y0 = std::max(0.0f, std::min(1.0f, y0));
                            y1 = std::max(0.0f, std::min(1.0f, y1));

                            BBox box;
                            box.score = score;
                            box.class_index = c;
                            box.x = x0;
                            box.y = y0;
                            box.w = x1 - x0;
                            box.h = y1 - y0;

                            all_boxes.push_back(box);
                        }
                    }
                }
            }
        }
    }
    
    // convert normalized coordinate to original image size
    float scale = std::max((float)img_h / net_h, (float)img_w / net_w);
    for (auto& bbox : all_boxes)
    {
        bbox.x = bbox.x * net_w * scale;
        bbox.y = bbox.y * net_h * scale;
        bbox.w = bbox.w * net_w * scale;
        bbox.h = bbox.h * net_h * scale;

        if (bbox.w < 1 || bbox.h < 1)
        {
            bbox.score = 0;
        }
        else
        {
            bbox.area = bbox.w * bbox.h;
        }
    }

    std::sort(all_boxes.begin(), all_boxes.end(), compareScore);

    // NMS
    std::vector<bool> excluded_boxes(all_boxes.size(), false);
    boxes.clear();

    for (int i = 0; i < all_boxes.size(); i++)
    {
        if (excluded_boxes[i] == true)
        {
            continue;
        }

        BBox& bbox_0 = all_boxes[i];
        if (bbox_0.score == 0)
        {
            continue;
        }

        boxes.push_back(bbox_0);
        int class_index_0 = bbox_0.class_index;

        for (int j = i + 1; j < all_boxes.size(); j++)
        {
            if (excluded_boxes[i] == true)
            {
                continue;
            }

            BBox& bbox_1 = all_boxes[j];
            if (bbox_1.class_index != class_index_0)
            {
                continue;
            }
            else if (bbox_1.score == 0)
            {
                continue;
            }
            else if (calcIOU(bbox_0, bbox_1) > iou_threshold)
            {
                excluded_boxes[j] = true;
            }
        }
    }
}

void DrawBox(std::vector<BBox>& boxes, cv::Mat& frame)
{
    if (boxes.size() == 0)
    {
        return;
    }

    for (auto& bbox : boxes)
    {
        std::ostringstream oss;
        oss << LABEL[bbox.class_index] << " " << std::fixed << std::setprecision(2) << bbox.score;
        std::string label = oss.str();

        std::vector<float> label_coords;
        std::string label_type;
        if (bbox.y > 20)
        {
            label_coords = { bbox.x, bbox.y };
            label_type = "top_outside";
        }
        else
        {
            label_coords = { bbox.x, bbox.y };
            label_type = "top_inside";
        }

        cv::rectangle(frame, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + bbox.w, bbox.y + bbox.h), cv::Scalar(0, 0, 0));

        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1.0, 3, 0);
        
        int padding = 5;
        int label_rect_w = textSize.width + padding * 2;
        int label_rect_h = textSize.height + padding * 2;

        if (label_type == "top_outside")
        {
            cv::rectangle(frame, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + label_rect_w, bbox.y - label_rect_h), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, label, cv::Point(bbox.x + padding, bbox.y - textSize.height + padding), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255));
        }
        else
        {
            cv::rectangle(frame, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + label_rect_w, bbox.y + label_rect_h), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, label, cv::Point(bbox.x + padding, bbox.y + textSize.height + padding), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255));
        }
         
    }
}