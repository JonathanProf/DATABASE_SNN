#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <string>
#include <fstream>
#include <iostream>
#define NUM_NEURONS 800

// Function prototypes
void getAssignments(unsigned short int *assignments, std::string fileName );
void getProportions(float *proportions, const int rows, const int cols, const std::string fileName );
void getWeights( float* weights, const std::string fileName, const int row, const int col );
void getTheta(float *theta, const std::string fileName );
void getInputSample(bool *arr, const std::string fileName, const int row, const int col );
float dotPointInputs(float *syn, bool *pixels_x_time, unsigned int neuronIndex );
float dotPointLayers(float *syn, bool *spikeVector);
int winner( bool *spikesE);
void setOneSpike(bool *spikes, int indexWin, unsigned short int *digits, unsigned short int *assignments);
void resetInputs(bool *spikesE, bool *spikesI);
void resetStateVariables();
int classification(unsigned short *spike_count, unsigned short *assignments);

#endif // FUNCTIONS_H
