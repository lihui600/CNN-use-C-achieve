#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cnn.h"

float activation_relu(float input, float bias)
{
	float temp = input + bias;
	return (temp > 0) ? temp : 0;
}

void MaxPooling(float** output, nSize outputSize, float** input, nSize inputSize, int kernel_size, int stride)
{
	int i, j, m, n;
	int t_m, t_n;
	t_m = t_n = 0;
	float sum;
	int count = 0;

	for (i = 0; i < outputSize.r; i++)
	{
		for (j = 0; j < outputSize.c; j++)
		{
			sum = 0.0;

			for (m = t_m; m < (t_m + kernel_size); m++)
			{
				for (n = t_n; n < (t_n + kernel_size); n++)
				{
					if (m > (inputSize.r - 1) || n > (inputSize.c - 1));
					else
					{
						if (sum < input[m][n])
						{
							sum = input[m][n];
						}
					}
				}
			}
			output[i][j] = sum;
			//printf("%f\n",output[i][j]);
			t_n = t_n + stride;
		}
		t_m = t_m + stride;
		t_n = 0;
	}
	//printf("n", count);
}



void read_file_conv(char* filename, int a, int b, int c, int d, CovLayer* conv)
{
	int count = 0;
	FILE* fp = fopen(filename, "rb");
	int i, j, m;
	for (i = 0; i < a; i++)
	{
		for (j = 0; j < b; j++)
		{
			for (m = 0; m < c; m++)
			{
				fread(conv->mapData[i][j][m], sizeof(float), d, fp);
				//printf("mapdata= %x\n",conv->mapData[i][j][m]);

			}

		}

	}
	/*for (int p = 0; p < a; p++)//6
	{
		for (int q = 0; q < b; q++)//1
		{
			for (int r = 0; r < c; r++)//5
			{
				for (int s = 0; s < d; s++)
				{
					printf("mapdata= %f\n", conv->mapData[p][q][r][s]);
					//count++;
				}
			}
		}
		printf("\n");
		count++;
	}
	printf("count=%d\n", count);
	*/
	fread(conv->basicData, sizeof(float), a, fp);
	fclose(fp);
}


void read_file_fc(char* filename, int a, int b, FcLayer* fc)
{
	FILE* fp = fopen(filename, "rb");
	int i, j;
	for (i = 0; i < a; i++)
	{
		fread(fc->wData[i], sizeof(float), b, fp);
	}
	fread(fc->basicData, sizeof(float), a, fp);
	
	fclose(fp);
}

void read_file_out(char* filename, int a, int b, OutLayer* out)
{
	FILE* fp = fopen(filename, "rb");
	int i, j;
	for (i = 0; i < a; i++)
	{
		fread(out->wData[i], sizeof(float), b, fp);
	}
	fread(out->basicData, sizeof(float), a, fp);
	for (i = 0; i < a; i++)
		for (j = 0; j < b; j++)
			printf("outL->wData[%d][%d]=%f\n", i, j, out->wData[i][j]);
	fclose(fp);
}

float** read_image(char* filename)
{
	FILE* file;
	file = fopen(filename, "rb");

	int w, h;
	fread(&w, sizeof(int), 1, file);
	fread(&h, sizeof(int), 1, file);
	fprintf(stderr, "image size w = %d, image size h =  %d \n", w, h);

	int i, j, k;

	float** image_data = (float**)malloc(sizeof(float*) * w);
	int temp[28];

	for (i = 0; i < w; i++)
	{
		image_data[i] = (float*)malloc(sizeof(float) * h);

		fread(temp, sizeof(float), h, file);
		for (j = 0; j < h; j++)
		{

			image_data[i][j] = temp[j] / 255.0;

			//printf("%f\n", image_data[i][j]);
		}
		//printf("\n");
	}
	//printf("\n");
	fclose(file);

	float** input_data;
	input_data = (float**)malloc(sizeof(float*));

	input_data = transpose_matrix(image_data, w, h);

	return input_data;
}

float*** input(float**inputData, char* Filename, int inputWidth, int inputHeight, int kernel_size, int inChannels, int outChannels)
{
	clock_t start, end;
	start = clock();

	int i, j, c, r;
	float** mapout;
	
	CovLayer* covL = (CovLayer*)malloc(sizeof(CovLayer));
	covL->mapData = (float****)malloc(outChannels * sizeof(float***));

	for (i = (outChannels - 1); i != (-1); i--)
	{
		covL->mapData[i] = (float***)malloc(inChannels * sizeof(float**));

		for (j = (inChannels - 1); j != (-1); j--)
		{
			covL->mapData[i][j] = (float**)malloc(kernel_size * sizeof(float*));

			for (r = (kernel_size - 1); r != (-1); r--)
			{
				covL->mapData[i][j][r] = (float*)malloc(kernel_size * sizeof(float));
				//intf("mapdata= %p\n", covL->mapData[i][j][r][c]);
			}
		}
	}
	covL->basicData = (float*)malloc(outChannels * sizeof(float));

	int outW = inputWidth - kernel_size + 1;
	int outH = inputHeight - kernel_size + 1;

	covL->v = (float***)malloc(sizeof(float**) * (outChannels));
	covL->y = (float***)malloc(sizeof(float**) * (outChannels));

	for (j = (outChannels - 1); j != (-1); j--)
	{
		covL->v[j] = (float**)malloc(sizeof(float*) * outH);
		covL->y[j] = (float**)malloc(sizeof(float*) * outH);
		for (r = 0; r < outH; r++)
		{
			covL->v[j][r] = (float*)calloc(outW, sizeof(float));
			covL->y[j][r] = (float*)calloc(outW, sizeof(float));
		}
	}

	read_file_conv(Filename, outChannels, inChannels, kernel_size, kernel_size, covL);

	nSize kernel_size1 = { kernel_size,kernel_size };
	nSize inSize = { inputWidth,inputHeight };
	nSize outSize = { outW,outH };


	for (i = 0; i < (outChannels); i++)
	{
		for (j = 0; j < (inChannels); j++)
		{
			mapout = cov(covL->mapData[i][j], kernel_size1, inputData, inSize, valid);
			addmat(covL->v[i], covL->v[i], outSize, mapout, outSize);
		}
		for (r = 0; r < outSize.r; r++)
		{
			for (c = 0; c < outSize.c; c++)
			{
				covL->y[i][r][c] = activation_relu(covL->v[i][r][c], covL->basicData[i]);
				//printf("%f\n", covL->y[i][r][c]);
			}
		}
		//printf("\n");
	}


	for (i = (outChannels - 1); i != (-1); i--)
	{
		for (j = (inChannels - 1); j != (-1); j--)
		{
			for (r = (kernel_size - 1); r != (-1); r--)
			{
				free(covL->mapData[i][j][r]);
			}
			free(covL->mapData[i][j]);
		}
		free(covL->mapData[i]);
	}
	free(covL->mapData);
	for (j = (outChannels - 1); j != (-1); j--)
	{
		for (r = 0; r < outH; r++)
		{
			free(covL->v[j][r]);
		}
		free(covL->v[j]);
	}
	free(covL->v);
	free(covL->basicData);

	end = clock();
	fprintf(stderr, "conv time = %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return covL->y;
}
float*** conv(float*** inputData, char* Filename, int inputWidth, int inputHeight, int kernel_size, int inChannels, int outChannels)
{
	clock_t start, end;
	start = clock();

	float** mapout;
	int i, j, c, r;

	CovLayer* covL = (CovLayer*)malloc(sizeof(CovLayer));//³õÊ¼»¯¾í»ý²ã
	covL->mapData = (float****)malloc(outChannels * sizeof(float***));

	for (i = (outChannels - 1); i != (-1); i--)
	{
		covL->mapData[i] = (float***)malloc(inChannels * sizeof(float**));

		for (j = (inChannels - 1); j != (-1); j--)
		{
			covL->mapData[i][j] = (float**)malloc(kernel_size * sizeof(float*));

			for (r = (kernel_size - 1); r != (-1); r--)
			{
				covL->mapData[i][j][r] = (float*)malloc(kernel_size * sizeof(float));
				//printf("mapdata= %p\n", covL->mapData[i][j][r][c]);
			}
		}
	}

	covL->basicData = (float*)malloc(outChannels * sizeof(float));
	
	int outW = inputWidth - kernel_size + 1;
	int outH = inputHeight - kernel_size + 1;
	covL->v = (float***)malloc(sizeof(float**) * (outChannels));
	covL->y = (float***)malloc(sizeof(float**) * (outChannels));
	for (j = (outChannels - 1); j != (-1); j--)
	{
		covL->v[j] = (float**)malloc(sizeof(float*) * outH);
		covL->y[j] = (float**)malloc(sizeof(float*) * outH);
		for (r = 0; r < outH; r++)
			//for (r = outH-1; r != (-1); r--)
		{
			covL->v[j][r] = (float*)calloc(outW, sizeof(float));
			covL->y[j][r] = (float*)calloc(outW, sizeof(float));
		}
	}


	read_file_conv(Filename, outChannels, inChannels, kernel_size, kernel_size, covL);


	nSize kernel_size1 = { kernel_size,kernel_size };
	nSize inSize = { inputWidth,inputHeight };
	nSize outSize = { outW,outH };


	for (i = 0; i < (outChannels); i++)
	{
		for (j = 0; j < (inChannels); j++)
		{
			mapout = cov(covL->mapData[i][j], kernel_size1, inputData[j], inSize, valid);
			addmat(covL->v[i], covL->v[i], outSize, mapout, outSize);	
		}
		for (r = 0; r < outSize.r; r++)
		{
			for (c = 0; c < outSize.c; c++)
			{
				covL->y[i][r][c] = activation_relu(covL->v[i][r][c], covL->basicData[i]);
				//printf("%f\n", covL->y[i][r][c]);
			}
		}
		//printf("\n");
	}


	for (i = (outChannels - 1); i != (-1); i--)
	{
		for (j = (inChannels - 1); j != (-1); j--)
		{
			for (r = (kernel_size - 1); r != (-1); r--)
			{
				free(covL->mapData[i][j][r]);	
			}
			free(covL->mapData[i][j]);
		}
		free(covL->mapData[i]);
	}
	free(covL->mapData);
	for (j = (outChannels - 1); j != (-1); j--)
	{
		for (r = 0; r < outH; r++)
		{
			free(covL->v[j][r]);
		}
		free(covL->v[j]);
	}
	free(covL->v);
	free(covL->basicData);

	end = clock();
	fprintf(stderr, "conv time = %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return covL->y;
}
float*** pool(float*** inputData, int inputWidth, int inputHeight, int poolType, int kernel_size, int inChannels, int outChannels, int stride)
{
	clock_t start, end;
	start = clock();

	int i;
	int count = 0;
	int outW, outH;
	if (stride == 2)
	{
		outW = ((inputWidth % 2 == 0) ? 0 : 1) + inputWidth / kernel_size;
		outH = ((inputHeight % 2 == 0) ? 0 : 1) + inputHeight / kernel_size;
	}
	else
	{
		outW = (inputWidth - kernel_size) / stride + 1;
		outH = (inputHeight - kernel_size) / stride + 1;
	}
	int outputWidth = outW;//12
	int outputHeight = outH;//12
	int j, r;

	PoolLayer* poolL = (PoolLayer*)malloc(sizeof(PoolLayer));
	poolL->y = (float***)malloc(outChannels * sizeof(float**));
	for (j = (outChannels - 1); j != (-1); j--)
	{
		poolL->y[j] = (float**)malloc(outH * sizeof(float*));
		for (r = (outH - 1); r != (-1); r--)
		{
			poolL->y[j][r] = (float*)calloc(outW, sizeof(float));
		}
	}

	nSize inSize = { inputWidth,inputHeight };
	nSize outSize = { outputWidth,outputHeight };

	for (i = 0; i < outChannels; i++)
	{
		MaxPooling(poolL->y[i], outSize, inputData[i], inSize, kernel_size, stride);
		//printf("	%f\n", poolL->y[i]);
		count++;
	}

	end = clock();
	fprintf(stderr, "pool time = %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return poolL->y;
}

float* flatten(float ***inputData,int inputWidth,int inputHeight,int inChannels,int outputNum)
{
	clock_t start, end;
	start = clock();

	int i,j,r,c;
	float temp = 0.0;

	FlLayer* Fl = (FlLayer*)malloc(sizeof(FlLayer));
	int w = inputWidth * inputHeight;
	Fl->v = (float**)malloc(inChannels * sizeof(float*));
	for (i = (inChannels - 1); i != (-1); i--)
	{
		Fl->v[i]=(float*)calloc(w,sizeof(float));
	}
	Fl->y = (float*)calloc(outputNum,sizeof(float));

	nSize inSize = { inputWidth,inputHeight };	
	int k = 0;
	int l = 0;

	for (j = 0; j < inChannels; j++)//16
	{
		
		for (r = 0; r < inSize.r; r++)
		{
			
			for (c = 0; c < inSize.c; c++)
			{
				Fl->v[j][l] = inputData[j][r][c];
				
				l++;
			}
		}
		l = 0;
	}


	for (int p = 0; p < inChannels; p++)
	{
		for (int d = p; d < w; d++)
		{
			temp = Fl->v[p][d];
			Fl->v[p][d] = Fl->v[d][p];
			Fl->v[d][p] = temp;
			
			
		}
	}
	for (int p = 0; p < inChannels; p++)
	{
		for (int d = 0; d < w; d++)
		{
			
			Fl->y[k] = Fl->v[p][d];
			//printf(" Fl->y[%d]= %f\n", k, Fl->y[k]);
			k++;
		}
	}

	for (i = (inChannels - 1); i != (-1); i--)
	{
		free(Fl->v[i]);
	}
	free(Fl->v);

	end = clock();
	fprintf(stderr, "fl time =         %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

	return Fl->y;
}
float* fc(float* inputData, char* filename, int inputNum, int outputNum)
{

	clock_t start, end;
	start = clock(); 

	int i, j, k;

	FcLayer* FcL = (FcLayer*)malloc(sizeof(FcLayer));
	FcL->wData = (float**)malloc(outputNum * sizeof(float*));
	for (i = (outputNum - 1); i != (-1); i--)
	{
		FcL->wData[i] = (float*)calloc(inputNum, sizeof(float));
	}
	FcL->basicData = (float*)calloc(outputNum, sizeof(float));
	FcL->y = (float*)calloc(outputNum, sizeof(float));


	read_file_fc(filename, outputNum, inputNum, FcL);
	

	for (i = 0; i < outputNum; ++i)
	{
		for (k = 0; k < inputNum; ++k)
		{
			FcL->y[i] = inputData[k] * FcL->wData[i][k] + FcL->y[i];
		}
		FcL->y[i] = activation_relu(FcL->y[i], FcL->basicData[i]);
		//printf("%f\n ", FcL->y[i]);
	}
	//printf("\n\n");


	for (i = (outputNum - 1); i != (-1); i--)
	{
		free(FcL->wData[i]);
	}
	free(FcL->wData);
	free(FcL->basicData);


	end = clock();
	fprintf(stderr, "fc time =         %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return FcL->y;
}


void* output(float* inputData, char* filename, int inputNum, int outputNum, char* filename1)
{
	clock_t start, end;
	start = clock();

	FILE* output = fopen(filename1, "w");
	int i, j, k;


	OutLayer* outL = (OutLayer*)malloc(sizeof(OutLayer));
	outL->wData = (float**)malloc(outputNum * sizeof(float*));
	
	for (i = (outputNum - 1); i != (-1); i--)
	{
		outL->wData[i] = (float*)calloc(inputNum,sizeof(float));
	}
	outL->basicData = (float*)calloc(outputNum, sizeof(float));
	outL->outputdata = (float*)calloc(outputNum, sizeof(float));

	read_file_fc(filename, outputNum, inputNum, outL);


	for (i = 0; i < outputNum; i++)
	{
		for (k = 0; k < inputNum; k++)
		{
			outL->outputdata[i] += inputData[k] * outL->wData[i][k];
		}
		//printf("%f\n ", cnn->fc160_1->y[i]);
		outL->outputdata[i] = activation_relu(outL->outputdata[i], outL->basicData[i]);
		//printf("%f\n ",outL->outputdata[i]);
		fprintf(output, "%f\n", outL->outputdata[i]);
	}
	fclose(output);


	for (i = (outputNum - 1); i != (-1); i--)
	{
		free(outL->wData[i]);
	}
	free(outL->wData);
	free(outL->basicData);


	end = clock();
	fprintf(stderr, "output_layer time =         %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

}

