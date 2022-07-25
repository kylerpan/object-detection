#pragma once

#include <opencv2/opencv.hpp>
#include "engine.h"

typedef struct tBBox
{
	float x, y, w, h;
	float area;
	int32_t class_index;
	float score;
} BBox;

void Postprocess(int32_t img_h, int32_t img_w, int32_t net_h, int32_t net_w, std::vector<Tensor>& tensors, std::vector<BBox>& boxes, float score_threshold = 0.5f, float iou_threshold = 0.5f);

void DrawBox(std::vector<BBox>& boxes, cv::Mat& frame);