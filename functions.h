#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <string>
#include <fstream>
#include <iostream>
#include <cassert>

void compute_Xe_Ae_spikes( const uint32_t  *input_sample,
                           uint16_t *spikesXePre,
                           const int t,
                           const uint16_t NUM_PIXELS
                           );
void update_exc_neurons( const uint16_t NUM_NEURONS,
                         const uint16_t NUM_PIXELS,
                         float *iSyn_Xe_Ae,
                         const uint16_t *spikesXePos,
                         const float *weightsXeAe,
                         float *iSyn_Ai_Ae,
                         const uint16_t *spikes_Ai_Ae_pos,
                         const float *weightsAiAe,
                         float *vE,
                         int *refrac_countE,
                         uint16_t *spikes_Ae_Ai_pre,
                         const float *theta
                         );

void update_inh_neurons( const uint16_t NUM_NEURONS,
                         float *vI,
                         const uint16_t *spikes_Ae_Ai_pos,
                         const float *weightsAeAi,
                         int *refrac_countI,
                         uint16_t *spikes_Ai_Ae_pre,
                         const uint16_t tamVector
                         );

void select_winner_neuron( const uint16_t NUM_NEURONS,
                           uint16_t *spikes_Ae_Ai_pre,
                           const int numSample,
                           const int t,
                           //uint16_t *indexArray,
                           const uint16_t* assignments,
                           const uint16_t SINGLE_SAMPLE_TIME,
                           unsigned short int* digits,
                           const uint16_t tamVector,
                           uint32_t *spike_count
                           );

void getWeights(float* weights,
                const std::string fileName,
                const uint32_t pixel,
                const uint32_t neuron
                );

void getTheta( float *theta,
               const std::string fileName,
               const uint32_t numNeurons
               );

void getInputSample( uint32_t *input,
                     const std::string fileName,
                     const int row,
                     const int col
                     );

void getAssignments( uint16_t *assignments,
                     std::string fileName,
                     const uint32_t numNeurons
                     );

#endif // FUNCTIONS_H
