// usage: ./cnn.app image_bin_filename
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cnn.h"

int main(int argc, char* argv[])
{
	clock_t start_all, end_all;
	start_all = clock();

	clock_t start, end;
	start = clock();

	char* filename;
	char* filename1;
	float** inputData;
	float*** y;
	float* y1;
	
	filename = "C:/model_weight/4.jpg.bin";
	inputData = read_image(filename);
	filename = "C:/model_weight/conv1.bin";
	y=input(inputData,filename,28,28,5,1,6);
	y=pool(y,24,24, MaxPool, 2,6,6,2);

	filename = "C:/model_weight/conv2.bin";
	y=conv(y,filename,12,12,5,6,16);
	y=pool(y,8,8, MaxPool,2,16,16,2);

	y1 = flatten(y, 4, 4, 16, 256);
	filename = "C:/model_weight/fc160_1.bin";
	y1= fc(y1,filename,256,120);

	filename = "C:/model_weight/fc160_2.bin";
	y1=fc(y1,filename,120,84);

	filename = "C:/model_weight/fc.bin";
	filename1 = "C:/model_weight/1.txt";
	output(y1, filename, 84, 10,filename1);


	end_all = clock();
	fprintf(stderr, "\n------------------------------------------\ntotal time          =  %f seconds\n------------------------------------------\n", (double)(end_all - start_all) / CLOCKS_PER_SEC);

	return 0;
}
