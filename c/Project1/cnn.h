#pragma once
//#ifndef __CNN_
//#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mat.h"
#define MaxPool 1

// conv layer
typedef struct convolutional_layer {
	

	//w is a 4d matrix, kernel_size*kernel_size*inChannels*outChannels*	
	float**** mapData;
	//b
	float* basicData;
	// the input of active function ‰»Îæÿ’Û
	float*** v;
	// the output of active function ‰≥ˆæÿ’Û
	float*** y;
}CovLayer;

// pooling layer
typedef struct pooling_layer {
	
	//the output data
	float*** y;
}PoolLayer;

// full connect layer
typedef struct fc_layer {
	

	float** wData; // w(inputNum*outputNum)
	float* basicData;   //b(outputNum)
	float* y;

}FcLayer;
typedef struct fl_layer {
	
	float** v;
	float* y;

}FlLayer;

// output layer
typedef struct out_layer {
	
	float** wData; // w(inputNum*outputNum)
	float* basicData;   //b(outputNum)
	float* outputdata;
}OutLayer;


float activation_relu(float input, float bas);
void MaxPooling(float** output, nSize outputSize, float** input, nSize inputSize, int kernel_size, int stride);
void read_file_fc(char* filename, int a, int b, FcLayer* fc);
void read_file_out(char* filename, int a, int b, OutLayer* fc);
void read_file_conv(char* filename, int a, int b, int c, int d, CovLayer* conv);
float** read_image(char* filename);
float*** input(float** inputData, char* Filename, int inputWidth, int inputHeight, int kernel_size, int inChannels, int outChannels);
float*** conv(float*** inputData, char* Filename, int inputWidth, int inputHeight, int kernel_size, int inChannels, int outChannels);
float*** pool(float*** inputData, int inputWidth, int inputHeight, int poolType, int kernel_size, int inChannels, int outChannels, int stride);
float* flatten(float*** inputData, int inputWeight, int inputHeight, int inChannels, int outputNum);
float* fc(float* inputData, char* filename, int inputNum, int outputNum);
void* output(float* inputData, char* filename, int inputNum, int outputNum, char* filename1);

