#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include "functions.h"

using namespace std;

#define NUM_NEURONS 400
#define NUM_PIXELS 784
#define SINGLE_SAMPLE_TIME 64
#define PATH_SAMPLES_POISSON "../DATABASE_SNN/inputSamples_64ms/%05d_inputSpikesPoisson_64ms.dat"
#define PATH_PARAMETERS_NET "../DATABASE_SNN/window64ms/BD400_64ms/"
#define PATH_RESULTS_NET "../DATABASE_SNN/classification/"
#define TOTAL_SAMPLES static_cast<int>(10000)

int main()
{
    assert( sizeof (uint32_t) == 4 );
    //! =====     =====     =====
    //! Variables Inicialization
    //! =====     =====     =====
    float vSyn = 0.0;

    float weights_Ae_Ai_constant = 22.5;
    float weights_Ai_Ae_constant = -120.0;

    float *theta;
    float *weightsXeAe; // connections Xe -> Ae
    float *weightsAeAi; // connections Ae -> Ai
    float *weightsAiAe; // connections Ae <- Ai
    uint32_t  *input_sample;

    // Check for spiking neurons
    bool *spikesXePre; // Spike occurrences Input
    bool *spikesXePos; // Spike occurrences Input
    bool spikes_Ae_Ai_pre[NUM_NEURONS] = {0};
    bool spikes_Ae_Ai_pos[NUM_NEURONS] = {0};
    bool spikes_Ai_Ae_pre[NUM_NEURONS] = {0};
    bool spikes_Ai_Ae_pos[NUM_NEURONS] = {0};

    unsigned short int *spike_count;
    unsigned short int *assignments;
    float *proportions;

    unsigned short int digits[10] = {0};

    float *vE;
    float *vI;

    int *refrac_countE;       // Refractory period counters
    int *refrac_countI;       // Refractory period counters
    int dt = 1;

    //! Constants definition
    float v_rest_e = -65.0;     // [mV]
    float v_reset_e = -60.0;    // [mV]
    float v_thresh_e = -52.0;   // [mV]
    int refrac_e = 5;           // [ms] Refractory time

    float v_rest_i = -60.0;     // [mV]
    float v_reset_i = -45.0;    // [mV]
    float v_thresh_i = -40.0;   // [mV]
    int refrac_i = 2;           // [ms] Refractory time

    //! [Step 1] Data structure initialization

    spikesXePre = new(std::nothrow) bool[NUM_PIXELS]{0};
    assert(spikesXePre != nullptr);

    spikesXePos = new(std::nothrow) bool[NUM_PIXELS]{0};
    assert(spikesXePos != nullptr);

    vE = new(std::nothrow) float[NUM_NEURONS]{0.0};
    assert( vE != nullptr );

    vI = new(std::nothrow) float[NUM_NEURONS]{0.0};
    assert( vI != nullptr );

    for (int indx = 0; indx < NUM_NEURONS; ++indx) {
        vE[indx] = v_rest_e;
        vI[indx] = v_rest_i;
    }

    refrac_countE = new(std::nothrow) int[NUM_NEURONS]{0};
    assert( refrac_countE != nullptr );

    refrac_countI = new(std::nothrow) int[NUM_NEURONS]{0};
    assert( refrac_countI != nullptr );

    unsigned int tamArr = SINGLE_SAMPLE_TIME/32;
    input_sample = new(std::nothrow) uint32_t[NUM_PIXELS * tamArr]{0};
    assert( input_sample != nullptr );

    spike_count = new(std::nothrow) unsigned short int[NUM_NEURONS]{0};
    assert( spike_count != nullptr );

    //! [Step 2] Loading data from files

    weightsXeAe = new(std::nothrow) float[NUM_PIXELS*NUM_NEURONS]{0};
    assert( weightsXeAe != nullptr );

    std::string filename = std::string(PATH_PARAMETERS_NET) + "XeAe.csv";
    getWeights(weightsXeAe, filename, NUM_PIXELS, NUM_NEURONS);

    weightsAeAi = new(std::nothrow) float[NUM_NEURONS*NUM_NEURONS]{0};
    assert( weightsAeAi != nullptr );

    for (int i = 0; i < NUM_NEURONS; ++i) {
        weightsAeAi[i*NUM_NEURONS+i] = weights_Ae_Ai_constant;
    }

    weightsAiAe = new(std::nothrow) float[NUM_NEURONS*NUM_NEURONS]{0};
    assert( weightsAiAe != nullptr );

    for (int i = 0; i < NUM_NEURONS; ++i) {
        for (int j = 0; j < NUM_NEURONS; ++j) {
            weightsAiAe[i*NUM_NEURONS+j] = (i != j) ? weights_Ai_Ae_constant : 0.0;
        }
    }

    //! theta values are loaded into the group of excitatory neurons
    theta = new(std::nothrow) float[NUM_NEURONS]{0};
    assert( theta != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "theta.csv";
    getTheta( theta, filename );

    assignments = new(std::nothrow) unsigned short int[NUM_NEURONS]{0};
    assert( assignments != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "assignments.csv";
    getAssignments( assignments, filename );

    proportions = new(std::nothrow) float[NUM_NEURONS*10]{0};
    assert( proportions != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "proportions.csv";
    getProportions( proportions, NUM_NEURONS, 10, filename );

    time_t start, end;
    time(&start);
    //! =====     =====     =====
    //! Run Simulation
    //! =====     =====     =====
    for (int numSample = 1; numSample <= TOTAL_SAMPLES; ++numSample) {

        char buffer[100];
        sprintf( buffer, PATH_SAMPLES_POISSON ,numSample);

        std::string filename(buffer);

        getInputSample( input_sample, filename, NUM_PIXELS, tamArr);

        //! Simulate network activity for SINGLE_SAMPLE_TIME timesteps.
        for (int t = 0; t < SINGLE_SAMPLE_TIME; ++t)
        {
            uint32_t datoAnalisis = 0;
            uint32_t resultOP = 0;
            uint8_t desplazamiento = 0;
            for (int j = 0; j < NUM_PIXELS; ++j)
            {
                // Los números pares serán el indice del pixel
                // La información del tiempo empieza desde el bit más significativo
                //  0 1
                //  2 3
                //  4 5
                //  6 7
                //  8 9
                // 10 11

                //!* Se desenvuelve el bucle ya que solo son dos posiciones del arreglo
                //input_sample[ j*2   ]
                //input_sample[ j*2+1 ]

                datoAnalisis = ( t < 32 ) ? input_sample[ j*2 ] : input_sample[ j*2+1 ];
                desplazamiento = ( t < 32 ) ? (31 - t) : (63 - t);
                resultOP = datoAnalisis & (1 << desplazamiento);
                resultOP >>= desplazamiento;

                // Se verifica que la información sea binaria
                assert( resultOP == 0 or resultOP == 1);
                spikesXePre[j] = resultOP;
                /*
                if ( input_sample[j+t*NUM_PIXELS] == 1 ) {
                    spikesXePre[j] = 1;
                }
                else{
                    spikesXePre[j] = 0;
                }
                */
            }


            // 1. Update E neurons
            for (unsigned int j = 0; j < NUM_NEURONS; ++j)
            {
                // Decay voltages and adaptive thresholds.
                vE[j] = 0.99f * (vE[j] - (-65.0f)) + (-65.0f);

                // Integrate inputs.
                vSyn = 0.0;

                for (unsigned int i = 0; i < NUM_NEURONS; ++i) {
                    vSyn += spikes_Ai_Ae_pos[i] ? weightsAiAe[j*NUM_NEURONS+i] : 0.0f ;
                }

                for (int i = 0; i < NUM_PIXELS; ++i) {
                    vSyn += spikesXePos[i] ? weightsXeAe[i*NUM_NEURONS+j] : 0.0f ;
                }

                if( refrac_countE[j] <= 0 )
                    vE[j] += vSyn;

                // Decrement refractory counters.
                refrac_countE[j] -= 1; // dt = 1

                // Check for spiking neurons.
                spikes_Ae_Ai_pre[j] = (vE[j] > ( (-52.0) + theta[j])) ? 1 : 0;

                // Refractoriness, voltage reset, and adaptive thresholds.
                if( spikes_Ae_Ai_pre[j] == 1 ){
                    refrac_countE[j] = 5; // refrac_e = 5;      // [ms]
                    vE[j] = -60.0F; // v_reset_e = -60.0;    // [mV]
                }

            }

            int indexWin = -1;
            for (int indx = 0; indx < NUM_NEURONS; ++indx) {
                if( spikes_Ae_Ai_pre[indx] != 0 ){
                    //cout << "[" << t << ", " << indx << "]" << endl;
                    indexWin = (indexWin == -1) ? indx : indexWin;
                }
            }

            if (indexWin >= 0) {

                for (int index = 0; index < NUM_NEURONS; ++index) {
                    if(spikes_Ae_Ai_pre[index]){
                        ++digits[ assignments[index] ];
                    }
                    spikes_Ae_Ai_pre[index] = (index == indexWin) ? 1 : 0;
                }
                ++spike_count[indexWin];
            }

            // 2. Update I neurons
            for (int i = 0; i < NUM_NEURONS; ++i) {
                // Decay voltages.
                vI[i] = 0.9048f * (vI[i] - (-60.0f)) + (-60.0f);

                // Integrate inputs.
                vSyn = 0.0;
                for (unsigned int j = 0; j < NUM_NEURONS; ++j) {
                    vSyn += spikes_Ae_Ai_pos[j] ? weightsAeAi[i*NUM_NEURONS+j] : 0.0f ;
                }

                if( refrac_countI[i] > 0 )
                    vSyn = 0;

                vI[i] += vSyn;

                // Decrement refractory counters.
                refrac_countI[i] -= 1;

                // Check for spiking neurons.
                if ( vI[i] > -40.0 ){
                    refrac_countI[i] =  2; // refrac_i = 2; // [ms]
                    vI[i] =  -45.0f; // v_reset_i = -45.0;    // [mV]
                    for (int j = 0; j < NUM_NEURONS; ++j) {
                        spikes_Ai_Ae_pre[j] = 0;
                    }
                    spikes_Ai_Ae_pre[i] = 1;
                }

            }

            // The arrays are exchanged for the next iteration of time
            for (int indx = 0; indx < NUM_PIXELS; ++indx) {
                spikesXePos[indx] = spikesXePre[indx];
            }

            for (int indx = 0; indx < NUM_NEURONS; ++indx) {
                spikes_Ae_Ai_pos[indx] = spikes_Ae_Ai_pre[indx];
                spikes_Ai_Ae_pos[indx] = spikes_Ai_Ae_pre[indx];
                spikes_Ae_Ai_pre[indx] = 0;
                spikes_Ai_Ae_pre[indx] = 0;
            }

        }

        // classification
        float rates[10] = {0.0f};

        //! Count the number of neurons with this label assignment.
        float n_assigns[10] = {0};

        for (int indx = 0; indx < NUM_NEURONS; ++indx) {
            ++n_assigns[assignments[indx]];

            rates[assignments[indx]] += spike_count[indx];
        }

        for (int indx = 0; indx < 10; ++indx) {
            rates[indx] /= n_assigns[indx];
        }

        int indWinner = 0;
        float ratesWin = 0;

        for (int indx = 0; indx < 10; ++indx) {
            if ( rates[indx] > ratesWin ) {
                indWinner = indx;
                ratesWin = rates[indx];
            }
        }

        std::cout << "Digit class: " << indWinner << std::endl;

        std::ofstream fileLabels;
        std::string filenameLabels = std::string(PATH_RESULTS_NET) + "labelsQt" + std::to_string(NUM_NEURONS) +"N_64ms.csv";
        fileLabels.open(filenameLabels, std::ofstream::out | std::ofstream::app);
        if (!fileLabels.is_open())
        {
            std::cout << "Error opening file" << __LINE__ << std::endl;
            exit(1);
        }

        fileLabels << indWinner << std::endl;
        fileLabels.close();

        //! =====     =====     =====
        //! Reset Variables
        //! =====     =====     =====
        for (int i = 0; i < NUM_NEURONS; ++i) {

            vE[i] = v_rest_e;
            vI[i] = v_rest_i;

            refrac_countE[i] = 0;
            refrac_countI[i] = 0;

            spike_count[i] = 0;

            spikes_Ae_Ai_pos[i] = 0;
            spikes_Ae_Ai_pre[i] = 0;
            spikes_Ai_Ae_pos[i] = 0;
            spikes_Ai_Ae_pre[i] = 0;
        }

        for (int i = 0; i < 784; ++i) {
            spikesXePre[i] = 0;
            spikesXePos[i] = 0;
        }
    }

    // Recording end time.
    time(&end);

    // Calculating total time taken by the program.
    double time_taken = double(end - start);
    cout << "Time taken by program is : " << time_taken << " sec" << endl;

    return 0;
}
